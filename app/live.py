"""Canlı yayın analizi: YouTube/HLS akışından örnek kareler alıp anlık istatistik üretir.

CPU'da kare başına ~1.2 s sürdüğü için her STRIDE karede bir örnek işlenir; bu da
saniyede yaklaşık bir örnek demektir. Akış gerçek zamanlı okunur, işlem yavaş
kaldığında aradaki kareler atlanır; böylece istatistik gecikmesi sınırlı kalır.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app import db, heatmap, pipeline, sports
from app.identity import PlayerIdentity
from sports.common.ball import BallTracker

STRIDE = max(1, int(os.environ.get("BAKIAPPI_LIVE_STRIDE", "30")))
IMGSZ = int(os.environ.get("BAKIAPPI_LIVE_IMGSZ", "960"))
SMOOTH, STEP = 7, 3
RECONNECT_ATTEMPTS = 60   # ~3 dk boyunca yeniden bağlanmayı dene
OPEN_TIMEOUT_MS = 10_000
READ_TIMEOUT_MS = 8_000
STALL_SECONDS = 25        # bu süre yeni örnek gelmezse yayın donmuş sayılır
WARMUP_LIMIT = 120
TEAM_FIT_SAMPLES = 24
STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://", "rtmps://")
RECORD_ENABLED = os.environ.get("BAKIAPPI_RECORD", "1") != "0"
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def resolve_stream(source: str) -> str:
    """Yerel dosya/RTSP aynen döner; YouTube vb. için oynatılabilir akış adresi çözülür."""
    if not source.lower().startswith(("http://", "https://")):
        return source
    if any(token in source for token in (".m3u8", ".mpd")):
        return source

    from yt_dlp import YoutubeDL

    with YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            # Canlı yayınlarda sesli birleşik format yok; HLS video-only gelir.
            "format": "bestvideo[height<=720]/bestvideo/best",
        }
    ) as ydl:
        info = ydl.extract_info(source, download=False)

    if info.get("url"):
        return info["url"]
    for fmt in reversed(info.get("formats") or []):
        if fmt.get("url"):
            return fmt["url"]
    raise RuntimeError("Yayın adresi çözülemedi; link canlı yayın mı?")


def _start_recording(match_id: str, url: str):
    """Yayını yeniden kodlamadan diske yazar; ffmpeg yoksa sessizce atlar.

    Kayıt ayrı bir ffmpeg sürecinde `-c copy` ile yapılır: CPU harcamaz ve
    analiz hattını yavaşlatmaz. Parçalı mp4 yazıldığı için süreç zorla
    kapansa bile dosya oynatılabilir kalır.
    """
    import shutil
    import subprocess

    if not RECORD_ENABLED or shutil.which("ffmpeg") is None:
        return None
    target = db.VIDEO_DIR / f"{match_id}.mp4"
    if target.exists():
        return None
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if url.lower().startswith("rtsp://"):
        command += ["-rtsp_transport", "tcp"]
    command += [
        "-i", url,
        "-map", "0:v:0",
        "-c", "copy",
        "-movflags", "+frag_keyframe+empty_moov",
        "-f", "mp4",
        str(target),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
    except OSError:
        return None


def _stop_recording(match: "LiveMatch") -> None:
    """Kaydı bitirir ve parçalı mp4'ü süre bilgisi olan mp4'e çevirir."""
    import subprocess

    recorder = getattr(match, "_recorder", None)
    if recorder is None:
        return
    match._recorder = None
    try:
        if recorder.stdin is not None:
            recorder.stdin.write(b"q")
            recorder.stdin.flush()
        recorder.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            recorder.kill()
        except Exception:  # noqa: BLE001
            pass

    # Parçalı mp4'ün süresi tarayıcıda 0 görünür; faststart'lı mp4'e remux et.
    path = db.VIDEO_DIR / f"{match.id}.mp4"
    if not path.is_file() or path.stat().st_size < 10_000:
        return
    temporary = path.with_name(f"{match.id}.remux.mp4")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-c", "copy", "-movflags", "+faststart",
                str(temporary),
            ],
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        temporary.unlink(missing_ok=True)
        return
    if completed.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 10_000:
        temporary.replace(path)
    else:
        temporary.unlink(missing_ok=True)


class CentroidTracker:
    """Seyrek örnekler için basit izleyici: hız tahmini + en yakın eşleşme.

    ByteTrack ardışık kare varsayar; canlı hatta kareler arasında saniye olduğu için
    kimlikler kopar. Bu izleyici hızı tahmin edip kapıyı ona göre açar.
    """

    def __init__(self, gate_ratio: float = 0.13, max_missing: int = 6) -> None:
        self.gate_ratio = gate_ratio
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}

    def _new(self, center: np.ndarray) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {"pos": center.copy(), "vel": np.zeros(2), "miss": 0}
        return track_id

    def update(self, centers: np.ndarray, width: int) -> np.ndarray:
        ids = np.full(len(centers), -1, dtype=int)
        track_ids = list(self.tracks)
        gate = self.gate_ratio * max(width, 320)

        if track_ids and len(centers):
            predicted = np.array(
                [self.tracks[tid]["pos"] + self.tracks[tid]["vel"] for tid in track_ids]
            )
            cost = np.linalg.norm(predicted[:, None, :] - centers[None, :, :], axis=2)
            candidates = sorted(
                (cost[r, c], r, c)
                for r in range(cost.shape[0])
                for c in range(cost.shape[1])
                if cost[r, c] <= gate
            )
            used_tracks, used_centers = set(), set()
            for _, row, column in candidates:
                if row in used_tracks or column in used_centers:
                    continue
                used_tracks.add(row)
                used_centers.add(column)
                track_id = track_ids[row]
                track = self.tracks[track_id]
                track["vel"] = 0.5 * track["vel"] + 0.5 * (
                    centers[column] - track["pos"]
                )
                track["pos"] = centers[column].copy()
                track["miss"] = 0
                ids[column] = track_id

        for column in range(len(centers)):
            if ids[column] == -1:
                ids[column] = self._new(centers[column])

        matched = set(ids.tolist())
        for track_id in track_ids:
            if track_id in matched:
                continue
            self.tracks[track_id]["miss"] += 1
            if self.tracks[track_id]["miss"] > self.max_missing:
                del self.tracks[track_id]

        return ids


class TeamResolver:
    """Takımı ve oyuncu görünüm kimliğini birlikte çözer.

    Isınma sırasında toplanan kırpımlarla takım sınıflandırıcısı bir kez
    eğitilir. Sonrasında her yeni tracker ID'si için takım ve görünüm gömlemi
    üretilir; kopan ID'ler aynı oyuncunun profiline bağlanır.
    """

    def __init__(self, device: str = "cpu", use_classifier: bool = True) -> None:
        self.device = device
        self.use_classifier = use_classifier
        self.classifier = None
        self.cache: Dict[int, int] = {}
        self.samples: list[np.ndarray] = []
        self.pending: Dict[int, np.ndarray] = {}
        self.identity = PlayerIdentity()

    def resolve(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        matrix: Optional[np.ndarray] = None,
        court_length_cm: Optional[float] = None,
    ) -> np.ndarray:
        if len(players) == 0:
            return np.zeros(0, dtype=int)

        # Isınma bitmeden takım saha yarısından tahmin edilir; sınıflandırıcı
        # hazır olduğunda aşağıdaki satırlar gerçek tahminle güncellenir.
        centers = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        if not self.use_classifier and matrix is not None:
            world = sports.project(matrix, centers)
            split = (court_length_cm or 1800.0) / 2.0
            teams = (world[:, 0] > split).astype(int)
        else:
            teams = (centers[:, 0] > frame.shape[1] / 2).astype(int)

        embeddings: list[Optional[np.ndarray]] = [None] * len(players)
        unseen: list[int] = []
        for row, tracker_id in enumerate(players.tracker_id):
            if tracker_id is None:
                continue
            key = int(tracker_id)
            if key in self.cache:
                teams[row] = self.cache[key]
            elif self.classifier is None:
                if self.use_classifier and len(self.samples) < 64:
                    crop = sv.crop_image(frame, players.xyxy[row])
                    if crop.size:
                        self.samples.append(crop)
                        self.pending[key] = crop
            else:
                unseen.append(row)

        if (
            self.use_classifier
            and self.classifier is None
            and len(self.samples) >= TEAM_FIT_SAMPLES
        ):
            self._fit()
            for row, tracker_id in enumerate(players.tracker_id):
                if tracker_id is not None:
                    teams[row] = self.cache.get(int(tracker_id), teams[row])

        if unseen:
            crops = [sv.crop_image(frame, players.xyxy[row]) for row in unseen]
            features = predicted = None
            try:
                features = self.classifier.extract_features(crops)
                predicted = self.classifier.predict_from_features(features)
            except Exception:  # noqa: BLE001
                pass
            if features is not None and predicted is not None:
                for position, row in enumerate(unseen):
                    key = int(players.tracker_id[row])
                    if position < len(predicted):
                        teams[row] = int(predicted[position])
                    if position < len(features):
                        embeddings[row] = features[position]
                    self.cache[key] = teams[row]

        self.identity.apply(players, embeddings, teams)
        return teams

    def _fit(self) -> None:
        try:
            from sports.common.team import TeamClassifier

            classifier = TeamClassifier(device=self.device)
            classifier.fit(self.samples)
        except Exception:  # noqa: BLE001
            return

        self.classifier = classifier
        try:
            keys = list(self.pending)
            features = classifier.extract_features(list(self.pending.values()))
            predicted = classifier.predict_from_features(features)
            for position, key in enumerate(keys):
                team = int(predicted[position]) if position < len(predicted) else 0
                self.cache[key] = team
                self.identity.enroll(key, features[position], team)
        except Exception:  # noqa: BLE001
            pass
        self.pending.clear()


def _clock(timestamp: float) -> str:
    return time.strftime("%d.%m.%Y %H:%M", time.localtime(timestamp))


class LiveMatch:
    """Tek bir canlı maçın arka plan işçisi ve anlık durumu."""

    def __init__(
        self,
        match_id: str,
        group_id: int,
        title: str,
        source: str,
        sport: str = "football",
        calibration: Optional[dict] = None,
    ) -> None:
        self.id = match_id
        self.group_id = group_id
        self.title = title
        self.source = source
        self.sport = sports.get(sport).key
        self.profile = sports.get(sport)
        self.calibration = calibration
        self._matrix = None
        if calibration:
            points = calibration.get("points") if isinstance(calibration, dict) else calibration
            extent = calibration.get("extent", "full") if isinstance(calibration, dict) else "full"
            try:
                self._matrix = sports.calibration_matrix(self.profile, points, str(extent))
            except ValueError:
                self._matrix = None
        self.ball_tracker = BallTracker() if self.profile.has_ball else None
        self.ball_frames = 0
        self.ball_heat: dict[int, int] = {}
        self.status = "connecting"
        self.stage = "yayın aranıyor"
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.players: Dict[int, dict] = {}
        self.player_heat: Dict[int, dict[int, int]] = {}
        self.team_heat: Dict[int, dict[int, int]] = {}
        self.samples = 0
        self.frames_read = 0
        self.noise_cm = 0.0
        self.team_source = "position"
        self.warm = True
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"live-{match_id}", daemon=True)
        self._last_sample_at = time.time()
        self._last_read_at = time.time()
        self.fps_source = 25.0
        self.is_stream = source.lower().startswith(STREAM_PREFIXES)
        self._finished = False
        self.paused = False
        self.pause_started: Optional[float] = None
        self.paused_total = 0.0
        self.period = 1
        self.period_log: list[dict] = []
        self._period_mark = 0.0
        self._resolver: Optional[TeamResolver] = None

    # ---------------------------------------------------------------- kontrol
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def elapsed_active(self, end: Optional[float] = None) -> float:
        """Duraklatılan süreler düşülmüş oyun süresi (saniye)."""
        stop = end if end is not None else (self.finished_at or time.time())
        paused = self.paused_total
        if self.paused and self.pause_started is not None:
            paused += max(0.0, stop - self.pause_started)
        return max(0.0, stop - self.started_at - paused)

    def set_paused(self, paused: bool) -> None:
        """Duraklatır/devam ettirir; duraklamada geçen süre istatistiğe sayılmaz."""
        with self.lock:
            if self.status not in {"live", "connecting"}:
                raise RuntimeError("Maç canlı değil.")
            if paused and not self.paused:
                self.paused = True
                self.pause_started = time.time()
            elif not paused and self.paused:
                if self.pause_started is not None:
                    self.paused_total += max(0.0, time.time() - self.pause_started)
                self.paused = False
                self.pause_started = None
                # Duraklama boyunca geçen süre örnek aralığına girmesin.
                self._last_sample_at = time.time()

    def set_period(self, period: int) -> None:
        """Devre/çeyrek numarasını değiştirir; kapanan bölümün süresi kaydedilir."""
        target = max(1, min(self.profile.period_count, int(period)))
        with self.lock:
            if target == self.period:
                return
            self._close_period()
            self.period = target

    def _close_period(self) -> None:
        elapsed = self.elapsed_active()
        self.period_log.append(
            {"devre": self.period, "sure_sn": round(max(0.0, elapsed - self._period_mark), 1)}
        )
        self._period_mark = elapsed

    def snapshot(self) -> dict:
        with self.lock:
            end = self.finished_at or time.time()
            elapsed = self.elapsed_active(end)
            paused_s = self.paused_total
            if self.paused and self.pause_started is not None:
                paused_s += max(0.0, end - self.pause_started)
            players = []
            for entry in self.players.values():
                # Ölçüm üretmemiş ya da yuvarlanınca 0 m kalan kısa izler tabloya girmesin.
                if entry.get("steps", 0) < 1 or entry["distance_m"] < 0.5:
                    continue
                seconds = max(1.0, entry["samples"] * max(entry.get("interval", 1.0), 0.2))
                zones = entry.get("zones") or {}
                run_m = float(zones.get("run", 0.0))
                sprint_m = float(zones.get("sprint", 0.0))
                players.append(
                    {
                        "tracker_id": entry["tracker_id"],
                        "team": entry["team"],
                        "distance_m": round(entry["distance_m"], 1),
                        "top_speed_kmh": round(entry["top_speed_kmh"], 1),
                        "avg_speed_kmh": round(entry["distance_m"] * 3.6 / seconds, 1),
                        "samples": entry["samples"],
                        "sprints": int(entry.get("sprints", 0)),
                        "high_speed_m": round(run_m + sprint_m, 1),
                        "zones": {
                            name: round(float(zones.get(name, 0.0)), 1)
                            for name in self.profile.zones
                        },
                    }
                )
            players.sort(key=lambda row: row["distance_m"], reverse=True)

            teams = {}
            for team in (0, 1):
                members = [row for row in players if row["team"] == team]
                teams[str(team)] = {
                    "players": len(members),
                    "distance_m": round(sum(row["distance_m"] for row in members), 1),
                    "top_speed_kmh": max([row["top_speed_kmh"] for row in members], default=0.0),
                }

            return {
                "id": self.id,
                "title": self.title,
                "sport": self.sport,
                "sport_name": self.profile.name,
                "periods_total": self.profile.period_count,
                "period_name": self.profile.period_name("tr"),
                "calibration": "manual" if self._matrix is not None else "none",
                "ball": {
                    "frames": self.ball_frames,
                    "visible_ratio": round(self.ball_frames / self.samples, 3)
                    if self.samples
                    else 0.0,
                }
                if self.profile.has_ball
                else None,
                "status": self.status,
                "stage": "duraklatıldı" if self.paused else self.stage,
                "paused": self.paused,
                "paused_s": round(paused_s, 1),
                "period": self.period,
                "periods": [dict(entry) for entry in self.period_log],
                "source_url": self.source,
                "started_at": self.started_at,
                "started_at_text": _clock(self.started_at),
                "finished_at": self.finished_at,
                "elapsed_s": round(elapsed, 1),
                "samples": self.samples,
                "frames_read": self.frames_read,
                "stride": STRIDE,
                "noise_cm": round(self.noise_cm, 1),
                "team_source": self.team_source,
                "id_repairs": self._resolver.identity.merge_count
                if self._resolver is not None
                else 0,
                "players": players,
                "teams": teams,
                "error": self.error,
            }

    def heat_snapshot(self) -> tuple[dict, dict, dict]:
        """Isı haritası sayılarının kopyası (oyuncu, takım, top)."""
        with self.lock:
            players = {key: dict(cells) for key, cells in self.player_heat.items()}
            teams = {key: dict(cells) for key, cells in self.team_heat.items()}
            ball = dict(self.ball_heat)
        return players, teams, ball

    def set_calibration(self, points, extent: str = "full") -> dict:
        """Çalışan maçta saha kalibrasyonunu günceller (canlıda anında uygulanır)."""
        matrix = sports.calibration_matrix(self.profile, points, extent)
        with self.lock:
            self._matrix = matrix
            self.calibration = {"points": points, "extent": extent}
            # Eski ısı hücreleri başka bir koordinat sistemine aitti; sıfırdan topla.
            self.player_heat.clear()
            self.team_heat.clear()
            self.ball_heat.clear()
            self._reset_motion = True
        return self.calibration

    def _stats_payload(self) -> dict:
        payload = self.snapshot()
        payload.pop("stage", None)
        return payload

    def _persist(self) -> None:
        db.update_match(self.id, stats=self._stats_payload(), status=self.status)

    # ------------------------------------------------------------------- işçi
    def run(self) -> None:
        db.update_match(self.id, status="live")
        self.status = "live"
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self.stage = "hata"
            self.finished_at = time.time()
            db.update_match(
                self.id,
                status="error",
                error=self.error,
                finished_at=self.finished_at,
                stats=self._stats_payload(),
            )
            return
        finally:
            _stop_recording(self)

        self.status = "finished"
        self.stage = "maç bitti"
        self.finished_at = time.time()
        self._finish()

    def _loop(self) -> None:
        self.stage = "modeller yükleniyor"
        profile = self.profile
        player_model = YOLO(str(profile.player_model)).to(device="cpu")
        pitch_model = (
            YOLO(str(profile.pitch_model)).to(device="cpu")
            if profile.pitch_model is not None and self._matrix is None
            else None
        )
        tracker = CentroidTracker()
        resolver = TeamResolver(use_classifier=profile.key != "volleyball")
        self._resolver = resolver
        motion: Optional[pipeline.MotionTracker] = None
        attempts = 0

        while not self.stop_event.is_set():
            try:
                url = resolve_stream(self.source)
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                if attempts > RECONNECT_ATTEMPTS:
                    raise RuntimeError(f"Yayın çözülemedi: {exc}") from exc
                self.stage = "yayın adresi alınamadı, yeniden deneniyor"
                self.stop_event.wait(3)
                continue

            capture = cv2.VideoCapture()
            opened = capture.open(
                url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    OPEN_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    READ_TIMEOUT_MS,
                ],
            )
            if not opened:
                capture.release()
                capture = cv2.VideoCapture(url)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                attempts += 1
                if attempts > RECONNECT_ATTEMPTS:
                    raise RuntimeError("Yayın akışı açılamadı.")
                self.stage = "yayına bağlanılamadı, yeniden deneniyor"
                self.stop_event.wait(3)
                continue

            if getattr(self, "_recorder", None) is None:
                self._recorder = _start_recording(self.id, url)

            attempts = 0
            self.stage = "canlı analiz"
            self.fps_source = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
            self._last_sample_at = time.time()
            self._last_read_at = time.time()

            while not self.stop_event.is_set():
                if self.paused and not self.is_stream:
                    # Dosyada kare kaybetmemek için duraklamada okuma da bekler.
                    self._last_sample_at = time.time()
                    self.stop_event.wait(0.2)
                    continue
                if self.is_stream and time.time() - self._last_sample_at > STALL_SECONDS:
                    self.stage = "yayın dondu"
                    break
                if not capture.grab():
                    break
                if self.is_stream:
                    # İşleme sürerken biriken kareleri atlayıp canlı uca yaklaş.
                    elapsed = time.time() - self._last_read_at
                    skip = int(elapsed * self.fps_source) - 1
                    skip = max(0, min(skip, int(self.fps_source * 5)))
                else:
                    skip = STRIDE - 1
                for _ in range(skip):
                    if not capture.grab():
                        break
                ok, frame = capture.retrieve()
                if not ok:
                    break
                self._last_read_at = time.time()
                if self.paused:
                    # Duraklatıldı: kare okunur ama istatistiğe yazılmaz.
                    self._last_sample_at = time.time()
                    continue
                self.frames_read += 1
                if getattr(self, "_reset_motion", False):
                    if motion is not None or self._matrix is not None:
                        motion = pipeline.MotionTracker(
                            1.0, smooth_frames=SMOOTH, step=STEP
                        )
                    self._reset_motion = False
                self._process(frame, player_model, pitch_model, tracker, resolver, motion)
                if motion is None and resolver.classifier is not None:
                    motion = pipeline.MotionTracker(1.0, smooth_frames=SMOOTH, step=STEP)
                    self.warm = False
                    self.team_source = "siglip"
                elif self.warm and self.samples >= WARMUP_LIMIT:
                    motion = pipeline.MotionTracker(1.0, smooth_frames=SMOOTH, step=STEP)
                    self.warm = False

            capture.release()
            if self.stop_event.is_set() or not self.is_stream:
                break
            self.stage = "yayın kesildi, yeniden bağlanıyor"
            self.stop_event.wait(2)

    def _process(self, frame, player_model, pitch_model, tracker, resolver, motion) -> None:
        now = time.time()
        profile = self.profile
        keypoints = None
        if pitch_model is not None:
            keypoints = sv.KeyPoints.from_ultralytics(pitch_model(frame, verbose=False)[0])
        detections = sv.Detections.from_ultralytics(
            player_model(frame, imgsz=IMGSZ, verbose=False)[0]
        )
        if len(detections):
            detections.tracker_id = tracker.update(
                detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER),
                frame.shape[1],
            )

        players = pipeline._class_detections(detections, profile.player_class)
        players = pipeline._on_court(players, self._matrix, profile)
        players_team = resolver.resolve(
            frame,
            players,
            self._matrix,
            profile.court_length_cm if profile.key == "volleyball" else None,
        )
        goalkeepers = pipeline._class_detections(detections, profile.goalkeeper_class)
        goalkeeper_team = pipeline.resolve_goalkeeper_teams(
            players, players_team, goalkeepers
        )
        referees = pipeline._class_detections(detections, profile.referee_class)
        merged, color_lookup = pipeline.merge_with_teams(
            players, players_team, goalkeepers, goalkeeper_team, referees
        )

        ball = None
        if self.ball_tracker is not None:
            ball = self.ball_tracker.update(
                pipeline._class_detections(detections, profile.ball_class)
            )
            if len(ball):
                self.ball_frames += 1

        self.samples += 1

        # Kalibrasyon varsa homografi sabittir; yoksa (futbol) anahtar noktalardan.
        matrix = self._matrix
        if matrix is None and keypoints is not None:
            matrix = pipeline.pitch_matrix(keypoints)
        if motion is not None and len(merged) > 0 and matrix is not None:
            if self.is_stream:
                per_sample = min(5.0, max(0.2, now - self._last_sample_at))
            else:
                per_sample = STRIDE / max(1.0, self.fps_source)
            # Ölçülen adım, STEP örnek aralığına denk gelir.
            motion.interval_s = STEP * per_sample
            image_xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            motion.update(matrix, image_xy, merged.tracker_id, color_lookup)
            self.noise_cm = motion.noise_cm

            with self.lock:
                for row, tracker_id in enumerate(merged.tracker_id):
                    if tracker_id is None or color_lookup[row] not in (0, 1):
                        continue
                    entry = self.players.setdefault(
                        int(tracker_id),
                        {
                            "tracker_id": int(tracker_id),
                            "team": int(color_lookup[row]),
                            "distance_m": 0.0,
                            "top_speed_kmh": 0.0,
                            "samples": 0,
                            "steps": 0,
                            "interval": per_sample,
                        },
                    )
                    entry["team"] = int(color_lookup[row])
                    entry["samples"] += 1
                    entry["interval"] = per_sample
                    if pipeline.record_motion(entry, motion, tracker_id, sport=profile):
                        entry["steps"] += 1

                # Isı haritası: düzeltilmiş saha konumu (cm) hücreye sayılır.
                for row, tracker_id in enumerate(merged.tracker_id):
                    if tracker_id is None or color_lookup[row] not in (0, 1):
                        continue
                    position = motion.corrected.get(int(tracker_id))
                    if position is None:
                        continue
                    heatmap.accumulate(
                        self.player_heat.setdefault(int(tracker_id), {}), position, profile
                    )
                    heatmap.accumulate(
                        self.team_heat.setdefault(int(color_lookup[row]), {}),
                        position,
                        profile,
                    )

                if ball is not None and len(ball):
                    centre = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    for position in pipeline.project(matrix, centre):
                        heatmap.accumulate(self.ball_heat, position, profile)

        self._last_sample_at = now
        if self.samples % 5 == 0:
            self._persist()

    # ------------------------------------------------------------------ bitiş
    def _finish(self) -> None:
        with self.lock:
            if self._finished:
                return
            self._finished = True
            self._close_period()
        payload = self._stats_payload()
        analysis = self._analysis(payload)
        folder = db.ANALYSIS_DIR / str(self.group_id)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime(self.started_at))
        path = folder / f"{stamp}-{self.id}.json"
        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        db.update_match(
            self.id,
            status="finished",
            finished_at=self.finished_at,
            analysis_path=str(path),
            stats=payload,
        )
        # Maç biter bitmez herkese açık bağlantı hazır olsun (paylaşım elle beklemesin).
        db.ensure_share_token(self.id)
        self._prebuild_cards(payload)

    def _prebuild_cards(self, payload: dict) -> None:
        """Maç biter bitmez oyuncu kartlarını arka planda hazırlar.

        Kart istenmeden önce hazır olurlar; hata olursa uç nokta isteğe bağlı
        üretmeye devam eder, bu yüzden sessizce yutulur.
        """

        def work() -> None:
            try:
                from app import cards

                # Maç bu arada silinmişse kart üretmeye hiç başlama.
                if db.get_match(self.id, self.group_id) is None:
                    return
                players = payload.get("players") or []
                awards = cards.match_awards(players)
                group = db.group_name(self.group_id)
                for player in players:
                    meta = cards.build_meta(
                        title=self.title,
                        date=_clock(self.started_at),
                        group=group,
                        match_id=self.id,
                        elapsed_s=payload.get("elapsed_s") or 0,
                        teams=payload.get("teams"),
                        team=int(player.get("team") or 0),
                        sport=getattr(self, "sport", "football"),
                    )
                    cards.save_card(
                        self.id,
                        player.get("tracker_id"),
                        "fi",
                        player,
                        meta,
                        awards.get(player.get("tracker_id"), []),
                    )
                # Üretim sırasında maç silindiyse yarım kalan kartları temizle.
                if db.get_match(self.id, self.group_id) is None:
                    cards.invalidate_match(self.id)
            except Exception:  # noqa: BLE001
                return

        threading.Thread(target=work, name=f"cards-{self.id}", daemon=True).start()

    def _analysis(self, payload: dict) -> dict:
        players = payload["players"]
        return {
            "mac": self.title,
            "tarih": _clock(self.started_at),
            "baslangic": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
            "bitis": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.finished_at or time.time())
            ),
            "sure_sn": payload["elapsed_s"],
            "duraklama_sn": payload.get("paused_s") or 0,
            "devreler": payload.get("periods") or [],
            "kaynak": self.source,
            "ornek_sayisi": payload["samples"],
            "takimlar": payload["teams"],
            "oyuncular": players,
            "isi_haritasi": {
                "hucre_m": self.profile.cell_m,
                "boyut": [self.profile.nx, self.profile.ny],
                "takimlar": {
                    str(team): {str(cell): count for cell, count in cells.items()}
                    for team, cells in self.team_heat.items()
                },
                "oyuncular": {
                    str(tracker_id): {str(cell): count for cell, count in cells.items()}
                    for tracker_id, cells in self.player_heat.items()
                },
                "top": {str(cell): count for cell, count in self.ball_heat.items()},
            },
            "en_cok_kosan": players[:3],
            "en_hizli": sorted(players, key=lambda row: -row["top_speed_kmh"])[:3],
            "brans": self.profile.name,
            "sport": self.sport,
            "period_name": self.profile.period_name("tr"),
            "top": {
                "frames": self.ball_frames,
                "visible_ratio": round(self.ball_frames / self.samples, 3) if self.samples else 0.0,
            }
            if self.profile.has_ball
            else None,
            "kalibrasyon": "manual"
            if self._matrix is not None
            else ("auto" if self.profile.pitch_model is not None else "none"),
            "not": (
                "Mesafeler saha homografisi ve canlı örnekleme ile yaklaşık hesaplanır; "
                "oyuncu kimlikleri izleyicinin verdiği numaralardır; görünüm "
                "benzerliğiyle onarılan kopmalar aynı numarada birleştirilir."
            ),
        }


MATCHES: Dict[str, LiveMatch] = {}
_lock = threading.Lock()


def start(
    match_id: str,
    group_id: int,
    title: str,
    source: str,
    sport: str = "football",
    calibration: Optional[dict] = None,
) -> LiveMatch:
    match = LiveMatch(match_id, group_id, title, source, sport, calibration)
    with _lock:
        MATCHES[match_id] = match
    match.start()
    return match


def get(match_id: str) -> Optional[LiveMatch]:
    with _lock:
        return MATCHES.get(match_id)


def stop(match_id: str) -> bool:
    match = get(match_id)
    if match is None:
        return False
    match.stop()
    return True


def forget_finished() -> None:
    """Bellekte biriken bitmiş maçları temizler."""
    with _lock:
        for match_id, match in list(MATCHES.items()):
            if match.status in {"finished", "error"} and not match.thread.is_alive():
                MATCHES.pop(match_id, None)
