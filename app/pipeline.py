"""BakIAppi — video analiz hattı.

RADAR görünümünü üretirken oyuncu/takım istatistiklerini toplar.

Mesafe ölçümü üç düzeltme içerir:
  1. Anahtar nokta homografisi yalnızca güveni yüksek noktalarla kurulur.
  2. Tüm oyuncuları aynı anda kaydıran sistematik bileşen, kare başına medyan
     akışla kestirilip hareketten çıkarılır.
  3. Bbox titreşiminden gelen konum gürültüsü ikinci farklardan ölçülür ve
     adım mesafesinden karesel olarak düşülür.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration

from app import sports
from app.identity import PlayerIdentity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "examples" / "soccer" / "data"
PLAYER_MODEL = DATA_DIR / "football-player-detection.pt"
PITCH_MODEL = DATA_DIR / "football-pitch-detection.pt"

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = SoccerPitchConfiguration()
VERTICES = np.array(CONFIG.vertices, dtype=np.float32)
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]

KEYPOINT_MIN_CONFIDENCE = 0.5
SMOOTH_FRAMES = 25      # mesafe ölçümünde medyan filtresi penceresi
SPEED_STEP = 12         # kaç karede bir mesafe ölçülür
MAX_PLAUSIBLE_SPEED_KMH = 30.0

# Hız bölgeleri (km/s): yürüme / jog / koşu / sprint
ZONE_WALK_KMH = 7.0
ZONE_JOG_KMH = 14.0
ZONE_RUN_KMH = 20.0
ZONES = ("walk", "jog", "run", "sprint")


def zone_of(speed_kmh: float, sport=None) -> str:
    """Hızı branşın bölgelerine göre sınıflandırır (futbol/basketbol eşikleri)."""
    return sports.get(sport).zone_of(speed_kmh)
MIN_TRACKS_FOR_SHIFT = 4  # ortak mod (kayma) tahmini için gereken en az iz sayısı

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Homografiyi noktalara uygular."""
    shaped = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2)


def pitch_matrix(keypoints: sv.KeyPoints) -> Optional[np.ndarray]:
    """Görüntü -> saha (cm) homografisi; güveni düşük anahtar noktalar atılır."""
    # Karede hiç saha anahtar noktası bulunamayabilir (yakın plan, bulanıklık).
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    xy = keypoints.xy[0].astype(np.float32)
    if keypoints.confidence is not None:
        mask = keypoints.confidence[0] >= KEYPOINT_MIN_CONFIDENCE
    else:
        mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if int(mask.sum()) < 4:
        return None
    matrix, _ = cv2.findHomography(xy[mask], VERTICES[mask])
    return matrix


class MotionTracker:
    """Oyuncuların saha konumlarını izler ve mesafeyi ölçer.

    Tüm oyuncuları aynı anda etkileyen sistematik bileşen (projeksiyon/kamera
    kayması) kare başına medyan akışla kestirilip hareketten çıkarılır. Mesafe,
    pencereye uydurulan doğrunun eğiminden alınır (tek karelik ID sıçramaları
    eğimi bozmaz); bbox titreşiminden gelen gürültü ise ikinci farklardan ölçülüp
    karesel olarak düşülür.
    """

    def __init__(
        self,
        fps: float,
        smooth_frames: int = SMOOTH_FRAMES,
        step: int = SPEED_STEP,
    ) -> None:
        self.fps = fps
        self.smooth = smooth_frames
        self.step = step
        self.interval_s = step / fps if fps else 0.5
        self.raw: Dict[int, np.ndarray] = {}
        self.corrected: Dict[int, np.ndarray] = {}
        self.history: Dict[int, Deque[np.ndarray]] = {}
        self.samples: Dict[int, int] = {}
        self.recent: Dict[int, np.ndarray] = {}
        self.shift: Optional[np.ndarray] = None
        self._second_sum = 0.0
        self._second_count = 0

    @property
    def noise_cm(self) -> float:
        """Kare başı konum gürültüsü (cm); beyaz gürültü varsayımıyla."""
        if self._second_count < 10:
            return 0.0
        return float(np.sqrt(self._second_sum / (6.0 * self._second_count)))

    def update(
        self,
        matrix: np.ndarray,
        image_points: np.ndarray,
        tracker_ids,
        color_lookup: np.ndarray,
    ) -> np.ndarray:
        """Karedeki noktaları saha koordinatına çevirir, düzeltilmiş konumu saklar."""
        world = project(matrix, image_points)

        visible = [
            row
            for row, tracker_id in enumerate(tracker_ids)
            if tracker_id is not None and color_lookup[row] in (0, 1)
        ]

        deltas: Dict[int, np.ndarray] = {}
        for row in visible:
            key = int(tracker_ids[row])
            previous = self.raw.get(key)
            if previous is not None:
                deltas[key] = world[row] - previous
            self.raw[key] = world[row].copy()

        # Kare için ortak (tüm izlerde aynı olan) kayma bileşeni.
        self.shift = None
        if len(deltas) >= MIN_TRACKS_FOR_SHIFT:
            self.shift = np.median(np.array(list(deltas.values())), axis=0)

        for row in visible:
            key = int(tracker_ids[row])
            current = self.corrected.get(key)
            delta = deltas.get(key)
            if current is None or delta is None:
                self.corrected[key] = world[row].copy()
            else:
                if self.shift is not None:
                    delta = delta - self.shift
                self.corrected[key] = current + delta
            self._track_noise(key, self.corrected[key])

        return world

    def _track_noise(self, key: int, position: np.ndarray) -> None:
        """İkinci farklarla konum gürültüsü varyansını biriktirir."""
        snapshot = self.recent.get(key)
        if snapshot is not None:
            second = position - 2 * snapshot[1] + snapshot[0]
            self._second_sum += float(np.dot(second, second))
            self._second_count += 1
            self.recent[key] = np.array([snapshot[1], position])
        else:
            self.recent[key] = np.array([position, position])

    def distance_step(self, tracker_id) -> Optional[float]:
        """Pencere dolduysa bu adımda kat edilen mesafeyi (cm) döndürür."""
        key = int(tracker_id)
        position = self.corrected.get(key)
        if position is None:
            return None

        self.samples[key] = self.samples.get(key, 0) + 1
        history = self.history.setdefault(key, deque(maxlen=self.smooth))
        history.append(position.astype(np.float32))
        if len(history) < self.smooth or self.samples[key] % self.step:
            return None

        samples = np.array(history, dtype=np.float32)

        # Pencereye doğru parçası uydur; hızı eğimden al. Tek karelik sıçrama
        # eğimi bozamaz, bu yüzden ID kopmaları hızı şişirmez.
        steps = np.arange(len(samples), dtype=np.float32)
        steps -= steps.mean()
        denominator = float((steps * steps).sum())
        if denominator <= 0:
            return None
        centered = samples - samples.mean(axis=0)
        slope = (steps[:, None] * centered).sum(axis=0) / denominator
        measured = float(np.linalg.norm(slope) * self.step)

        # Hareket doğruya yakın olmalı; ID takasında konum sıçrar ve artık büyür.
        # Eşik geniş tutulur: gerçek dönüşler de artık üretir.
        predicted = samples.mean(axis=0) + steps[:, None] * slope
        residual = np.linalg.norm(samples - predicted, axis=1)
        if float(residual.max()) > 900.0:
            return None

        # Eğim kestiriminin gürültü varyansı sigma^2 / sum(t^2); adım vektörü 2 boyutlu.
        sigma = self.noise_cm
        sigma_step2 = 2.0 * (self.step**2) * sigma**2 / denominator
        return float(np.sqrt(max(0.0, measured**2 - sigma_step2)))


def resolve_player_teams(
    frame: np.ndarray,
    players: sv.Detections,
    classifier: Optional[TeamClassifier],
    cache: Dict[int, int],
    profile=None,
    matrix: Optional[np.ndarray] = None,
    court_length_cm: Optional[float] = None,
) -> tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """Takım kimliği ve satır başına görünüm gömlemi üretir.

    Gömlem yalnızca ilk kez görülen track_id için hesaplanır; kimlik onarımı
    da tam olarak o anda yeni bir ID'nin sahibini aradığı için yeterlidir.
    """
    embeddings: List[Optional[np.ndarray]] = [None] * len(players)
    if len(players) == 0:
        return np.zeros(0, dtype=int), embeddings

    if profile is not None and profile.key == "volleyball":
        # Voleybolda takımlar fileyle ayrılır; forma kümelemesi yerine kort
        # koordinatındaki taraf kullanılır. Bu, yedek/seyirci gürültüsünden
        # etkilenmez ve 6-6 dağılımını korur.
        centers = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        if matrix is not None:
            world = sports.project(matrix, centers)
            split = (court_length_cm or profile.court_length_cm) / 2.0
            return (world[:, 0] > split).astype(int), embeddings
        return (centers[:, 0] > frame.shape[1] / 2).astype(int), embeddings

    if classifier is None:
        centers = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return (centers[:, 0] > frame.shape[1] / 2).astype(int), embeddings

    team_ids = np.zeros(len(players), dtype=int)
    pending_rows: List[int] = []
    pending_crops: List[np.ndarray] = []

    for row, tracker_id in enumerate(players.tracker_id):
        if tracker_id is not None and int(tracker_id) in cache:
            team_ids[row] = cache[int(tracker_id)]
        else:
            pending_rows.append(row)
            pending_crops.append(sv.crop_image(frame, players.xyxy[row]))

    if pending_crops:
        features = None
        try:
            features = classifier.extract_features(pending_crops)
            predicted = classifier.predict_from_features(features)
        except Exception:  # noqa: BLE001
            predicted = np.zeros(len(pending_crops), dtype=int)
        for position, row in enumerate(pending_rows):
            team_ids[row] = int(predicted[position]) if position < len(predicted) else 0
            if features is not None and position < len(features):
                embeddings[row] = features[position]
            tracker_id = players.tracker_id[row]
            if tracker_id is not None:
                cache[int(tracker_id)] = team_ids[row]

    return team_ids, embeddings


def resolve_goalkeeper_teams(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """Kalecileri en yakın takım merkezine atar."""
    if len(goalkeepers) == 0 or len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalkeeper_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    centroids = np.array(
        [
            players_xy[players_team_id == team].mean(axis=0)
            if (players_team_id == team).any()
            else players_xy.mean(axis=0)
            for team in (0, 1)
        ]
    )
    distances = np.linalg.norm(goalkeeper_xy[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1).astype(int)


def merge_with_teams(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
    goalkeepers_team_id: np.ndarray,
    referees: sv.Detections,
) -> tuple[sv.Detections, np.ndarray]:
    """Oyuncu, kaleci ve hakemleri tek listede birleştirir; renk indeksini üretir."""
    merged = sv.Detections.merge([players, goalkeepers, referees])
    color_lookup = np.array(
        players_team_id.tolist()
        + goalkeepers_team_id.tolist()
        + [REFEREE_CLASS_ID] * len(referees)
    )
    return merged, color_lookup


def annotate_players(
    frame: np.ndarray, detections: sv.Detections, color_lookup: np.ndarray
) -> np.ndarray:
    """Tespitleri takım rengiyle elips + numara olarak çizer."""
    annotated = ELLIPSE_ANNOTATOR.annotate(
        frame, detections, custom_color_lookup=color_lookup
    )
    labels = [str(t) if t is not None else "-" for t in detections.tracker_id]
    return ELLIPSE_LABEL_ANNOTATOR.annotate(
        annotated, detections, labels, custom_color_lookup=color_lookup
    )


def record_motion(
    entry: dict,
    motion: "MotionTracker",
    tracker_id,
    interval_s: Optional[float] = None,
    sport=None,
) -> bool:
    """Bir oyuncunun bu adımdaki mesafe/hız katkısını istatistiğe ekler.

    Ölçüm kabul edildiyse True döndürür; kısa ömürlü izler hiç adım üretmez.
    Hız bölgeleri ve makul hız sınırı branşa göre değişir.
    """
    profile = sports.get(sport)
    step_cm = motion.distance_step(tracker_id)
    if step_cm is None:
        return False
    interval = interval_s if interval_s is not None else motion.interval_s
    speed = step_cm / max(interval, 1e-3) * 0.036
    if speed >= profile.max_speed_kmh:
        return False
    distance_m = step_cm / 100.0
    zone = profile.zone_of(speed)
    zones = entry.setdefault("zones", {name: 0.0 for name in profile.zones})
    zones[zone] = zones.get(zone, 0.0) + distance_m
    if zone == "sprint" and entry.get("last_zone") != "sprint":
        entry["sprints"] = entry.get("sprints", 0) + 1
    entry["last_zone"] = zone

    entry["distance_m"] += distance_m
    entry["top_speed_kmh"] = max(entry["top_speed_kmh"], round(speed, 1))
    return True


def draw_radar(
    frame: np.ndarray, pitch_xy: np.ndarray, color_lookup: np.ndarray, sport=None
) -> np.ndarray:
    """Saha/kort görünümünü çizip karenin alt ortasına bindirir."""
    profile = sports.get(sport)
    height, width, _ = frame.shape

    if profile.key == "football":
        radar = draw_pitch(config=CONFIG)
        for team, color in enumerate(COLORS):
            mask = color_lookup == team
            if not mask.any():
                continue
            radar = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_xy[mask],
                face_color=sv.Color.from_hex(color),
                radius=20,
                pitch=radar,
            )
    else:
        radar = sports.radar_bgr(profile, width // 2, height // 2, padding=8)
        radar_h, radar_w, _ = radar.shape
        scale_x = (radar_w - 16) / profile.court_length_cm
        scale_y = (radar_h - 16) / profile.court_width_cm
        for row, (x, y) in enumerate(pitch_xy):
            if not (0 <= x < profile.court_length_cm and 0 <= y < profile.court_width_cm):
                continue
            color = sv.Color.from_hex(COLORS[int(color_lookup[row]) % len(COLORS)]).as_bgr()
            cv2.circle(
                radar,
                (int(8 + x * scale_x), int(8 + y * scale_y)),
                4,
                color,
                -1,
            )

    radar = sv.resize_image(radar, (width // 2, height // 2))
    radar_h, radar_w, _ = radar.shape
    rect = sv.Rect(
        x=width // 2 - radar_w // 2,
        y=height - radar_h,
        width=radar_w,
        height=radar_h,
    )
    return sv.draw_image(frame, radar, opacity=0.5, rect=rect)


def _on_court(
    players: sv.Detections,
    matrix: Optional[np.ndarray],
    profile: sports.SportProfile,
    serve_zone_cm: float = 300.0,
    sideline_cm: float = 30.0,
) -> sv.Detections:
    """Voleybolda saha dışındaki yedek, seyirci ve masa görevlilerini eler.

    Servis atan oyuncu arka çizginin dışında durabilir; bu yüzden arka çizgi
    yönünde 3 m'lik servis alanı bırakılır. Yan çizgide ise yalnızca küçük bir
    tolerans vardır; yedek oyuncular ve skor masası bu sayede dışarıda kalır.
    """
    if len(players) == 0 or matrix is None or profile.key != "volleyball":
        return players
    anchors = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    world = sports.project(matrix, anchors)
    mask = (
        (world[:, 0] >= -serve_zone_cm)
        & (world[:, 0] <= profile.court_length_cm + serve_zone_cm)
        & (world[:, 1] >= -sideline_cm)
        & (world[:, 1] <= profile.court_width_cm + sideline_cm)
    )
    return players[mask]


def fit_team_classifier(
    player_model,
    source_path: str,
    total_frames: int,
    imgsz: int,
    device: str,
    sport=None,
    matrix: Optional[np.ndarray] = None,
) -> Optional[TeamClassifier]:
    """Videodan örnek kırpımlar toplayıp takım sınıflandırıcısını eğitir."""
    profile = sports.get(sport)
    if profile.key == "volleyball":
        return None
    crops: List[np.ndarray] = []
    stride = max(1, total_frames // 20)
    for frame in sv.get_video_frames_generator(source_path, stride=stride, end=total_frames):
        result = player_model(frame, imgsz=imgsz, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is None:
            continue
        players = _class_detections(detections, profile.player_class)
        players = _on_court(players, matrix, profile)
        crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]

    if len(crops) < 8:
        return None
    try:
        classifier = TeamClassifier(device=device)
        classifier.fit(crops)
        return classifier
    except Exception:  # noqa: BLE001
        return None


def _class_detections(detections: sv.Detections, class_id: Optional[int]) -> sv.Detections:
    """Bir sınıfın tespitlerini döndürür; sınıf yoksa boş küme verir."""
    mask = np.zeros(len(detections), dtype=bool)
    if class_id is not None and detections.class_id is not None:
        mask = detections.class_id == class_id
    return detections[mask]


def analyze(
    source_path: str,
    target_path: str,
    max_seconds: float = 0.0,
    imgsz: int = 1280,
    device: str = "cpu",
    progress: Optional[Callable[[dict], None]] = None,
    sport: str = "football",
    calibration: Optional[dict] = None,
) -> dict:
    """Videoyu işler, çizimli videoyu yazar ve istatistikleri döndürür.

    ``calibration`` verilirse (görüntüdeki dört kort köşesi) homografi anahtar
    nokta modeli yerine bu noktalardan kurulur; basketbol bunu zorunlu kullanır.
    """
    source_path, target_path = str(source_path), str(target_path)
    profile = sports.get(sport)
    started = time.time()

    info = sv.VideoInfo.from_video_path(source_path)
    fps = float(info.fps or 25.0)
    total_frames = int(info.total_frames)
    if max_seconds and max_seconds > 0:
        total_frames = min(total_frames, int(round(fps * float(max_seconds))))
    total_frames = max(1, total_frames)

    def report(**kwargs) -> None:
        if progress is not None:
            progress(kwargs)

    report(stage="modeller yükleniyor", percent=0.0, frames_done=0, frames_total=total_frames)
    player_model = YOLO(str(profile.player_model)).to(device=device)
    pitch_model = None
    if profile.pitch_model is not None and calibration is None:
        pitch_model = YOLO(str(profile.pitch_model)).to(device=device)

    fixed_matrix = None
    calibration_extent = "full"
    if calibration:
        points = calibration.get("points") if isinstance(calibration, dict) else calibration
        calibration_extent = calibration.get("extent", "full") if isinstance(calibration, dict) else "full"
        fixed_matrix = sports.calibration_matrix(profile, points, str(calibration_extent))
    court_length_cm = profile.court_length_cm / (2.0 if calibration_extent == "half" else 1.0)

    report(stage="takım modeli hazırlanıyor", percent=1.0, frames_done=0, frames_total=total_frames)
    classifier = fit_team_classifier(
        player_model, source_path, total_frames, imgsz, device, profile, fixed_matrix
    )

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    motion = MotionTracker(fps)
    team_cache: Dict[int, int] = {}
    identity = PlayerIdentity()
    player_stats: Dict[int, dict] = {}
    players_per_frame: List[int] = []
    ball_tracker = BallTracker() if profile.has_ball else None
    ball_frames = 0

    frames_done = 0
    last_report = 0.0

    with sv.VideoSink(target_path, info) as sink:
        for frame in sv.get_video_frames_generator(source_path, end=total_frames):
            frames_done += 1

            keypoints = None
            if pitch_model is not None:
                keypoints = sv.KeyPoints.from_ultralytics(pitch_model(frame, verbose=False)[0])
            detections = sv.Detections.from_ultralytics(
                player_model(frame, imgsz=imgsz, verbose=False)[0]
            )
            detections = tracker.update_with_detections(detections)

            players = _class_detections(detections, profile.player_class)
            players = _on_court(players, fixed_matrix, profile)
            players_team_id, embeddings = resolve_player_teams(
                frame,
                players,
                classifier,
                team_cache,
                profile,
                fixed_matrix,
                court_length_cm,
            )
            identity.apply(players, embeddings, players_team_id)

            goalkeepers = _class_detections(detections, profile.goalkeeper_class)
            goalkeepers_team_id = resolve_goalkeeper_teams(
                players, players_team_id, goalkeepers
            )
            referees = _class_detections(detections, profile.referee_class)

            merged, color_lookup = merge_with_teams(
                players, players_team_id, goalkeepers, goalkeepers_team_id, referees
            )
            players_per_frame.append(len(players) + len(goalkeepers))

            annotated = annotate_players(frame.copy(), merged, color_lookup)

            ball = None
            if ball_tracker is not None:
                ball = ball_tracker.update(_class_detections(detections, profile.ball_class))
                if len(ball):
                    ball_frames += 1
                    for x, y in ball.get_anchors_coordinates(sv.Position.CENTER):
                        cv2.circle(annotated, (int(x), int(y)), 7, (0, 215, 255), -1)

            matrix = fixed_matrix
            if matrix is None and keypoints is not None:
                matrix = pitch_matrix(keypoints)
            if matrix is not None and len(merged) > 0:
                image_xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_xy = motion.update(matrix, image_xy, merged.tracker_id, color_lookup)
                annotated = draw_radar(annotated, pitch_xy, color_lookup, profile)

                for row, tracker_id in enumerate(merged.tracker_id):
                    if tracker_id is None or color_lookup[row] not in (0, 1):
                        continue
                    entry = player_stats.setdefault(
                        int(tracker_id),
                        {
                            "tracker_id": int(tracker_id),
                            "team": int(color_lookup[row]),
                            "distance_m": 0.0,
                            "top_speed_kmh": 0.0,
                            "frames": 0,
                            "zones": {name: 0.0 for name in profile.zones},
                        },
                    )
                    entry["frames"] += 1
                    record_motion(entry, motion, tracker_id, sport=profile)

            sink.write_frame(annotated)

            now = time.time()
            if now - last_report > 0.4 or frames_done == total_frames:
                last_report = now
                report(
                    stage="analiz ediliyor",
                    percent=min(99.0, frames_done / total_frames * 100.0),
                    frames_done=frames_done,
                    frames_total=total_frames,
                    elapsed_s=round(now - started, 1),
                )

    # Ölçüm penceresini dolduramayan kısa izler istatistiğe girmesin.
    measured = [player for player in player_stats.values() if player["frames"] >= SMOOTH_FRAMES]
    players_sorted = sorted(measured, key=lambda player: player["distance_m"], reverse=True)
    for entry in players_sorted:
        entry["distance_m"] = round(entry["distance_m"], 1)
        seconds = max(1.0, entry["frames"] / fps)
        entry["avg_speed_kmh"] = round(entry["distance_m"] * 3.6 / seconds, 1)

    teams = {}
    for team in (0, 1):
        members = [player for player in players_sorted if player["team"] == team]
        teams[str(team)] = {
            "players": len(members),
            "distance_m": round(sum(player["distance_m"] for player in members), 1),
            "top_speed_kmh": max([player["top_speed_kmh"] for player in members], default=0.0),
        }

    stats = {
        "source": Path(source_path).name,
        "sport": profile.key,
        "sport_name": profile.name,
        "periods": profile.period_count,
        "period_name": profile.period_name("tr"),
        "fps": fps,
        "frames_processed": frames_done,
        "duration_s": round(frames_done / fps, 1),
        "resolution": f"{info.width}x{info.height}",
        "elapsed_s": round(time.time() - started, 1),
        "players": {
            "tracked": len(players_sorted),
            "avg_per_frame": round(float(np.mean(players_per_frame)), 1)
            if players_per_frame
            else 0.0,
            "max_per_frame": max(players_per_frame, default=0),
        },
        "ball": {
            "frames": ball_frames,
            "visible_ratio": round(ball_frames / frames_done, 3) if frames_done else 0.0,
        }
        if profile.has_ball
        else None,
        "teams": teams,
        "noise_cm": round(motion.noise_cm, 1),
        "team_source": "siglip" if classifier is not None else "position",
        "calibration": "manual"
        if fixed_matrix is not None
        else ("keypoints" if pitch_model is not None else "none"),
        "id_repairs": identity.merge_count,
        "player_rows": players_sorted[:60],
    }

    report(
        stage="tamamlandı",
        percent=100.0,
        frames_done=frames_done,
        frames_total=total_frames,
        elapsed_s=stats["elapsed_s"],
    )
    return stats
