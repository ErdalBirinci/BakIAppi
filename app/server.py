"""BakIAppi — grup hesapları, canlı yayın analizi ve video analizi (FastAPI)."""

from __future__ import annotations

import csv
import io
import json
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import cards, db, heatmap, live, pipeline, sports

BASE = Path(__file__).resolve().parent
JOBS_DIR = BASE / "data" / "jobs"
REPORTS_DIR = BASE / "data" / "reports"
UPLOADS_DIR = BASE / "data" / "uploads"
FRAMES_DIR = BASE / "data" / "frames"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

SERVER_STARTED_AT = time.time()
COOKIE_NAME = "bakiappi_session"
SESSION_SECONDS = db.SESSION_TTL

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Sunucu yeniden başladıysa yarıda kalan canlı maçlar "live" kalmasın.

    Bu iş import anında değil, uygulama gerçekten ayağa kalkarken yapılır; böylece
    portu tutamayıp ölen ikinci bir süreç canlı maçları yanlışlıkla kesmez.
    """
    cards.prune_stale()
    db.prune_sessions()
    # Yarıda kalan maçları hata saymak yerine kaldığı yerden devam ettirmeyi dene.
    for match in db.live_matches(started_before=SERVER_STARTED_AT):
        calibration = None
        if match.get("calibration"):
            try:
                calibration = json.loads(match["calibration"])
            except json.JSONDecodeError:
                calibration = None
        live.start(
            match["id"],
            match["group_id"],
            match["title"],
            match["source_url"],
            match.get("sport") or "football",
            calibration,
        )
    yield


app = FastAPI(title="BakIAppi", lifespan=lifespan)

JOBS: Dict[str, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ oturumlar
class RegisterPayload(BaseModel):
    group_name: str
    username: str
    password: str


class LoginPayload(BaseModel):
    username: str
    password: str


class NamesPayload(BaseModel):
    names: Dict[str, str] = {}


class MatchPayload(BaseModel):
    title: str = ""
    source_url: str
    sport: Optional[str] = None
    calibration: Optional[dict] = None


class SportPayload(BaseModel):
    sport: str


class CalibrationPayload(BaseModel):
    points: list
    extent: str = "full"


class ControlPayload(BaseModel):
    paused: Optional[bool] = None
    period: Optional[int] = None


class EventPayload(BaseModel):
    tracker_id: Optional[int] = None
    kind: str
    minute: Optional[float] = None


class RemapPayload(BaseModel):
    source: int
    target: int
    mode: str = "swap"


class RenamePlayerPayload(BaseModel):
    name: str


def current_group(request: Request) -> dict:
    group = db.group_for_token(request.cookies.get(COOKIE_NAME))
    if group is None:
        raise HTTPException(status_code=401, detail="Giriş gerekli.")
    return group


def set_session(response: Response, group_id: int) -> None:
    response.set_cookie(
        COOKIE_NAME,
        db.create_session(group_id),
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )


@app.post("/api/auth/register")
def register(payload: RegisterPayload, response: Response) -> dict:
    name = payload.group_name.strip()
    username = payload.username.strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Grup adı en az 2 karakter olmalı.")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Kullanıcı adı en az 3 karakter olmalı.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı.")
    try:
        group_id = db.create_group(name, username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    set_session(response, group_id)
    return {"id": group_id, "name": name, "username": username}


LOGIN_WINDOW_S = 300.0
LOGIN_MAX_ATTEMPTS = 8
_login_hits: Dict[str, list] = {}
_login_lock = threading.Lock()


def _login_key(request: Request, username: str) -> str:
    host = request.client.host if request.client else "?"
    return f"{host}:{username.strip().casefold()}"


def _login_blocked(key: str) -> bool:
    cutoff = time.time() - LOGIN_WINDOW_S
    with _login_lock:
        stamps = [stamp for stamp in _login_hits.get(key, []) if stamp > cutoff]
        if stamps:
            _login_hits[key] = stamps
        else:
            _login_hits.pop(key, None)
        return len(stamps) >= LOGIN_MAX_ATTEMPTS


def _login_failed(key: str) -> None:
    now = time.time()
    with _login_lock:
        _login_hits.setdefault(key, []).append(now)
        for stale in [k for k, stamps in _login_hits.items() if not stamps or stamps[-1] < now - LOGIN_WINDOW_S]:
            _login_hits.pop(stale, None)


def _login_clear(key: str) -> None:
    with _login_lock:
        _login_hits.pop(key, None)


@app.post("/api/auth/login")
def login(payload: LoginPayload, request: Request, response: Response) -> dict:
    key = _login_key(request, payload.username)
    if _login_blocked(key):
        raise HTTPException(
            status_code=429,
            detail="Çok fazla hatalı deneme. Birkaç dakika sonra tekrar dene.",
        )
    group = db.verify_login(payload.username, payload.password)
    if group is None:
        _login_failed(key)
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı.")
    _login_clear(key)
    set_session(response, group["id"])
    return group


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db.delete_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request) -> dict:
    return current_group(request)


@app.get("/api/sports")
def list_sports(lang: str = "tr") -> dict:
    """Branş seçimi kartları: saha ölçüsü, periyot ve kalibrasyon yöntemi."""
    return {"sports": sports.payload(lang)}


@app.put("/api/sport")
def set_sport(payload: SportPayload, request: Request) -> dict:
    """Grubun branşını kaydeder; panel ve varsayılan maç branşı bunu izler."""
    group = current_group(request)
    if not sports.is_valid(payload.sport):
        raise HTTPException(status_code=400, detail="Geçersiz branş.")
    profile = sports.get(payload.sport)
    db.set_group_sport(group["id"], profile.key)
    return {"sport": profile.key, "name": profile.name}


# ------------------------------------------------------------------ maçlar
VIDEO_DIR = db.VIDEO_DIR
CLIPS_DIR = VIDEO_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _video_file(match_id: str) -> Optional[Path]:
    """Kaydedilmiş maç videosu varsa yolunu verir."""
    path = VIDEO_DIR / f"{match_id}.mp4"
    try:
        if path.is_file() and path.stat().st_size > 10_000:
            return path
    except OSError:
        pass
    return None


def _capture_frame(match: dict, position_s: float = 0.0) -> Path:
    """Maçın kaydından ya da canlı yayınından bir kare JPEG olarak çıkarır.

    Kalibrasyon arayüzü bu kareyi gösterip dört kort köşesinin tıklanmasını ister.
    """
    target = FRAMES_DIR / f"{match['id']}.jpg"
    position = max(0.0, min(float(position_s), 6 * 3600))
    if position <= 0.5 and target.is_file() and time.time() - target.stat().st_mtime < 120:
        return target

    source = _video_file(match["id"])
    capture = cv2.VideoCapture()
    try:
        if source is not None:
            capture.open(str(source))
            if position:
                capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
        else:
            url = live.resolve_stream(match["source_url"])
            capture.open(
                url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    live.OPEN_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    live.READ_TIMEOUT_MS,
                ],
            )
        ok, frame = capture.read() if capture.isOpened() else (False, None)
    except Exception:  # noqa: BLE001
        ok, frame = False, None
    finally:
        capture.release()
    if not ok or frame is None:
        raise HTTPException(status_code=404, detail="Kare alınamadı; yayın henüz hazır olmayabilir.")
    height, width = frame.shape[:2]
    if width > 1600:
        scale = 1600.0 / width
        frame = cv2.resize(frame, (1600, int(height * scale)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise HTTPException(status_code=500, detail="Kare kodlanamadı.")
    target.write_bytes(encoded.tobytes())
    return target


def _owned_video(match_id: str, request: Request) -> Path:
    """Sahiplik kontrolüyle maç videosunu verir."""
    group = current_group(request)
    if db.get_match(match_id, group["id"]) is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    path = _video_file(match_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Bu maçın videosu kaydedilmemiş.")
    return path


def match_payload(match: dict) -> dict:
    """Canlı maçta bellekdeki taze durumu veritabanı kaydının önüne koyar."""
    snapshot = live.get(match["id"])
    if snapshot is not None and match["status"] in {"live", "connecting"}:
        fresh = snapshot.snapshot()
        payload = {
            **match,
            "status": fresh["status"],
            "stats": fresh,
            "stage": fresh["stage"],
            "error": fresh["error"],
        }
    else:
        payload = dict(match)
    payload["video"] = _video_file(match["id"]) is not None
    return payload


@app.post("/api/matches")
def create_match(payload: MatchPayload, request: Request) -> dict:
    group = current_group(request)
    source = payload.source_url.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Yayın linki gerekli.")
    if not source.lower().startswith(live.STREAM_PREFIXES):
        candidate = Path(source)
        if not candidate.is_file():
            raise HTTPException(
                status_code=400,
                detail="Geçerli bir yayın adresi (YouTube / RTSP / RTMP) ya da yerel dosya yolu gir.",
            )
        source = str(candidate.resolve())

    if payload.sport is not None and not sports.is_valid(payload.sport):
        raise HTTPException(status_code=400, detail="Geçersiz branş.")
    profile = sports.get(payload.sport or group.get("sport") or sports.DEFAULT)
    calibration = payload.calibration or None
    if calibration:
        try:
            sports.calibration_matrix(
                profile, calibration.get("points"), calibration.get("extent", "full")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    default_title = "Karşılaşma" if profile.key in {"basketball", "volleyball"} else "Maç"
    title = payload.title.strip() or f"{default_title} {time.strftime('%d.%m.%Y %H:%M')}"
    match_id = uuid.uuid4().hex[:12]
    db.create_match(
        match_id,
        group["id"],
        title,
        source,
        sport=profile.key,
        calibration=json.dumps(calibration, ensure_ascii=False) if calibration else None,
    )
    live.start(match_id, group["id"], title, source, profile.key, calibration)
    return {"id": match_id, "title": title, "sport": profile.key}


@app.get("/api/matches")
def list_matches(request: Request) -> list:
    group = current_group(request)
    live.forget_finished()
    selected = group.get("sport") or sports.DEFAULT
    return [match_payload(match) for match in db.list_matches(group["id"], sport=selected)]


@app.get("/api/matches/{match_id}")
def get_match(match_id: str, request: Request) -> dict:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    payload = match_payload(match)
    profile = sports.get(match.get("sport"))
    if profile.key == "volleyball":
        # Canlı ekran da set numarasını aynı kaynaktan okusun; ayrı sayaç yok.
        payload["sets"] = _sets_state(profile, db.list_events(match_id))
    return payload


@app.get("/api/matches/{match_id}/frame.jpg")
def match_frame(match_id: str, request: Request, t: float = 0) -> FileResponse:
    """Kalibrasyon için maçtan bir kare gösterir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    path = _capture_frame(match, t)
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@app.get("/api/matches/{match_id}/calibration")
def get_calibration(match_id: str, request: Request) -> dict:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    profile = sports.get(match.get("sport"))
    calibration = match.get("calibration") or {}
    extent = calibration.get("extent", "full") if isinstance(calibration, dict) else "full"
    return {
        "sport": profile.key,
        "calibration": calibration or None,
        "court_cm": sports.court_corners(profile, extent),
        "method": "manual" if profile.pitch_model is None else "auto",
    }


@app.put("/api/matches/{match_id}/calibration")
def set_calibration(
    match_id: str, payload: CalibrationPayload, request: Request
) -> dict:
    """Dört kort köşesini kaydeder; canlı maçta anında uygulanır."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    profile = sports.get(match.get("sport"))
    extent = "half" if str(payload.extent).lower() == "half" else "full"
    try:
        sports.calibration_matrix(profile, payload.points, extent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    calibration = {"points": payload.points, "extent": extent}
    running = live.get(match_id)
    if running is not None:
        running.set_calibration(payload.points, extent)
    db.update_match(match_id, calibration=json.dumps(calibration, ensure_ascii=False))
    # Eski koordinat sistemine ait ısı haritası ve kartlar yeniden üretilsin.
    shutil.rmtree(heatmap.CACHE_DIR / match_id, ignore_errors=True)
    cards.invalidate_match(match_id)
    return {"ok": True, "calibration": calibration}


@app.get("/api/matches/{match_id}/video")
def match_video(match_id: str, request: Request) -> FileResponse:
    """Canlı maçta kaydedilen tam maç videosunu verir."""
    return FileResponse(
        _owned_video(match_id, request),
        media_type="video/mp4",
        filename=f"{match_id}.mp4",
    )


@app.get("/api/matches/{match_id}/video/clip")
def match_video_clip(
    match_id: str, request: Request, start: float = 0, duration: float = 20
) -> FileResponse:
    """Maç videosundan yeniden kodlamadan klip keser (anahtar kareye hizalar)."""
    source = _owned_video(match_id, request)
    start = max(0.0, min(float(start), 6 * 3600))
    duration = max(1.0, min(float(duration), 180.0))
    target = CLIPS_DIR / f"{match_id}-{int(start)}-{int(duration)}.mp4"
    if not target.is_file():
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.1f}", "-i", str(source),
                "-t", f"{duration:.1f}", "-c", "copy",
                "-movflags", "+faststart", str(target),
            ],
            capture_output=True,
            timeout=180,
        )
        if completed.returncode != 0 or not target.is_file():
            raise HTTPException(status_code=500, detail="Klip oluşturulamadı.")
    return FileResponse(
        target, media_type="video/mp4", filename=f"{match_id}-{int(start)}s.mp4"
    )


@app.post("/api/matches/{match_id}/control")
def control_match(match_id: str, payload: ControlPayload, request: Request) -> dict:
    """Canlı maçı duraklatır/devam ettirir ve devre numarasını değiştirir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    if payload.paused is None and payload.period is None:
        raise HTTPException(status_code=400, detail="Değişiklik belirtilmedi.")

    running = live.get(match_id)
    if running is None or match["status"] not in {"live", "connecting"}:
        raise HTTPException(status_code=409, detail="Maç canlı değil.")
    try:
        if payload.paused is not None:
            running.set_paused(bool(payload.paused))
        if payload.period is not None:
            running.set_period(int(payload.period))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    stats = running.snapshot()
    stats.pop("stage", None)
    db.update_match(match_id, stats=stats)
    return match_payload(db.get_match(match_id, group["id"]))


@app.post("/api/matches/{match_id}/stop")
def stop_match(match_id: str, request: Request) -> dict:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    live.stop(match_id)
    return {"ok": True}


@app.post("/api/matches/{match_id}/reconnect")
def reconnect_match(match_id: str, request: Request) -> dict:
    """Hataya düşen canlı maçı aynı kaynakla yeniden başlatır."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    if match["status"] != "error":
        raise HTTPException(status_code=409, detail="Yalnızca hataya düşen maç yeniden bağlanabilir.")

    live.forget_finished()
    db.update_match(match_id, status="live", error=None, finished_at=None)
    live.start(
        match_id,
        group["id"],
        match["title"],
        match["source_url"],
        match.get("sport") or "football",
        match.get("calibration"),
    )
    return {"ok": True}


def _invalidate_outputs(match_id: str) -> None:
    """Maçın önbelleklenmiş kart ve takım raporlarını siler."""
    cards.invalidate_match(match_id)
    for report in REPORTS_DIR.glob(f"{match_id}-*.png"):
        report.unlink(missing_ok=True)


@app.delete("/api/matches/{match_id}")
def delete_match(match_id: str, request: Request) -> dict:
    """Maçı, olayları, isimleri ve üretilen dosyalarıyla birlikte siler."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    running = live.get(match_id)
    if running is not None:
        # İşçi dururken analizi yazar; dosya ancak o bitince oluşur.
        running.stop()
        running.thread.join(timeout=15)

    if not db.delete_match(match_id, group["id"]):
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    _invalidate_outputs(match_id)
    shutil.rmtree(heatmap.CACHE_DIR / match_id, ignore_errors=True)
    for analysis in db.ANALYSIS_DIR.rglob(f"*-{match_id}.json"):
        analysis.unlink(missing_ok=True)
    (db.VIDEO_DIR / f"{match_id}.mp4").unlink(missing_ok=True)
    (FRAMES_DIR / f"{match_id}.jpg").unlink(missing_ok=True)
    for clip in (db.VIDEO_DIR / "clips").glob(f"{match_id}-*.mp4"):
        clip.unlink(missing_ok=True)
    live.forget_finished()
    return {"ok": True}


@app.patch("/api/players/{player_id}")
def rename_player(player_id: int, payload: RenamePlayerPayload, request: Request) -> dict:
    """Kadro oyuncusunu yeniden adlandırır; geçmiş maçlar da yeni ismi gösterir."""
    group = current_group(request)
    try:
        player = db.rename_player(group["id"], player_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for match in db.list_matches(group["id"]):
        _invalidate_outputs(match["id"])
    return player


_ANALYSIS_CACHE: dict = {}


def _analysis_path(match: dict, *, required: bool = True) -> Optional[Path]:
    """Analiz dosyasının yolunu çözer; `required` değilse eksikse None döner."""
    if not match.get("analysis_path"):
        if required:
            raise HTTPException(status_code=404, detail="Maç analizi henüz hazır değil.")
        return None
    path = Path(match["analysis_path"])
    if path.is_file():
        return path
    if required:
        raise HTTPException(status_code=404, detail="Analiz dosyası bulunamadı.")
    return None


def _read_analysis(path: Path) -> dict:
    """Analiz JSON'unu yol + değişiklik zamanına göre önbellekle okur."""
    key = (str(path), path.stat().st_mtime)
    cached = _ANALYSIS_CACHE.get(key)
    if cached is not None:
        return cached
    analysis = json.loads(path.read_text(encoding="utf-8"))
    if len(_ANALYSIS_CACHE) > 200:
        _ANALYSIS_CACHE.clear()
    _ANALYSIS_CACHE[key] = analysis
    return analysis


def _analysis_of(match: dict, *, missing_ok: bool = False) -> Optional[dict]:
    """Bir maçın analizini okur. `missing_ok` ise hazır olmayan analiz None döner."""
    path = _analysis_path(match, required=not missing_ok)
    if path is None:
        return None
    try:
        return _read_analysis(path)
    except (OSError, json.JSONDecodeError):
        if missing_ok:
            return None
        raise HTTPException(status_code=500, detail="Analiz dosyası okunamadı.")


@app.get("/api/matches/{match_id}/analysis")
def get_analysis(match_id: str, request: Request) -> dict:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return _analysis_of(match)


@app.get("/api/matches/{match_id}/players/{tracker_id}/card.png")
def player_card(
    match_id: str,
    tracker_id: int,
    request: Request,
    lang: str = "fi",
    download: int = 0,
) -> FileResponse:
    """Maç sonu oyuncu kartını 1080×1350 PNG olarak üretir (baytlar diskte saklanır)."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    analysis = _analysis_of(match)
    players = analysis.get("oyuncular") or []
    player = next(
        (row for row in players if int(row.get("tracker_id", -1)) == tracker_id), None
    )
    if player is None:
        raise HTTPException(status_code=404, detail="Oyuncu bulunamadı.")

    name = db.match_names(match_id).get(tracker_id)
    if name:
        player = {**player, "name": name}

    # Kart önbellekteyse ısı toplama/çizimi hiç yapılmaz.
    path = cards.card_path(match_id, tracker_id, lang)
    if not path.is_file():
        meta = cards.build_meta(
            title=match.get("title") or analysis.get("mac") or "",
            date=analysis.get("tarih") or "",
            group=group["name"],
            match_id=match_id,
            elapsed_s=analysis.get("sure_sn") or 0,
            teams=analysis.get("takimlar"),
            team=int(player.get("team") or 0),
            sport=match.get("sport"),
        )
        try:
            player_heat, _, _ = _heat_buckets(match)
        except HTTPException:
            player_heat = {}
        cells = player_heat.get(tracker_id) or {}
        cards.save_card(
            match_id,
            tracker_id,
            lang,
            player,
            meta,
            cards.match_awards(players).get(tracker_id, []),
            heat_png=heatmap.render(cells, match.get("sport")),
            heat_box=heatmap.focus_box(cells, sport=match.get("sport")),
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"bakiappi-{match_id}-{tracker_id}.png" if download else None,
        headers={"Cache-Control": "private, no-cache"},
    )


@app.get("/api/matches/{match_id}/names")
def get_names(match_id: str, request: Request) -> dict:
    group = current_group(request)
    if db.get_match(match_id, group["id"]) is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return {"names": {str(key): name for key, name in db.match_names(match_id).items()}}


@app.put("/api/matches/{match_id}/names")
def set_names(match_id: str, payload: NamesPayload, request: Request) -> dict:
    """Maç sonunda oyunculara isim atar; kart önbelleği temizlenir."""
    group = current_group(request)
    if db.get_match(match_id, group["id"]) is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    names: dict[int, str] = {}
    for key, value in payload.names.items():
        try:
            tracker_id = int(key)
        except (TypeError, ValueError):
            continue
        name = str(value).strip()[: db.MAX_NAME_LENGTH]
        if name:
            names[tracker_id] = name

    db.set_match_names(match_id, group["id"], names)
    _invalidate_outputs(match_id)
    return {"names": {str(key): name for key, name in names.items()}}


def _match_players(match: dict) -> list[dict]:
    """Canlı maçta bellekten, bitmiş maçta analiz dosyasından oyuncu satırları."""
    running = live.get(match["id"])
    if running is not None and match["status"] in {"live", "connecting"}:
        return running.snapshot().get("players") or []
    return _analysis_of(match).get("oyuncular") or []


REPORT_LABELS = {
    "tr": {
        "score": "Skor",
        "goals": "Goller",
        "baskets": "Basketler",
        "points": "Sayılar",
        "sets": "Setler",
        "current_set": "Devam eden set",
        "errors": "Hatalar (rakip sayı)",
        "assists": "Asistler",
        "teams": "Takım özeti",
        "runners": "En çok koşanlar",
        "fastest": "En hızlı",
        "none": "Gol/asist kaydı yok.",
        "none_volleyball": "Sayı/olay kaydı yok.",
        "players": "oyuncu",
        "note": "Mesafe ve hızlar saha homografisiyle yaklaşık hesaplanır.",
    },
    "fi": {
        "score": "Tilanne",
        "goals": "Maalit",
        "baskets": "Korit",
        "points": "Pisteet",
        "sets": "Erät",
        "current_set": "Käynnissä oleva erä",
        "errors": "Virheet (vastustajalle piste)",
        "assists": "Syötöt",
        "teams": "Joukkueiden yhteenveto",
        "runners": "Eniten juosseet",
        "fastest": "Nopein",
        "none": "Maali- tai syöttömerkintöjä ei ole.",
        "none_volleyball": "Piste- tai tapahtumamerkintöjä ei ole.",
        "players": "pelaajaa",
        "note": "Matkat ja nopeudet lasketaan likimäärin kentän homografialla.",
    },
    "en": {
        "score": "Score",
        "goals": "Goals",
        "baskets": "Baskets",
        "points": "Points",
        "sets": "Sets",
        "current_set": "Current set",
        "errors": "Errors (opponent point)",
        "assists": "Assists",
        "teams": "Team summary",
        "runners": "Top distance",
        "fastest": "Fastest",
        "none": "No goals or assists recorded.",
        "none_volleyball": "No points or events recorded.",
        "players": "players",
        "note": "Distances and speeds are approximate, computed from the pitch homography.",
    },
}


def _team_letter(team) -> str:
    try:
        return "A" if int(team) == 0 else "B"
    except (TypeError, ValueError):
        return "?"


def _report_text(match: dict, lang: str) -> str:
    """Analizden okunabilir bir maç raporu üretir (gol, asist, koşu, hız)."""
    labels = REPORT_LABELS.get(lang, REPORT_LABELS["tr"])
    analysis = _analysis_of(match)
    names = db.match_names(match["id"])
    events = _events_with_names(match["id"])
    rows = list(analysis.get("oyuncular") or [])
    teams = analysis.get("takimlar") or {}
    profile = sports.get(match.get("sport"))
    goals = [event for event in events if event.get("kind") in profile.event_points]
    assists = [event for event in events if event.get("kind") == "assist"]
    errors = [event for event in goals if event.get("kind") in profile.error_kinds]
    scored = [event for event in goals if event.get("kind") not in profile.error_kinds]
    score = {0: 0, 1: 0}
    for event in goals:
        try:
            score[sports.scoring_team(profile, event)] += profile.event_points.get(
                event.get("kind"), 1
            )
        except (KeyError, TypeError, ValueError):
            pass
    if profile.key == "basketball":
        scoring_label = labels["baskets"]
    elif profile.key == "volleyball":
        scoring_label = labels["points"]
    else:
        scoring_label = labels["goals"]

    stamp = time.strftime("%d.%m.%Y %H:%M", time.localtime(match["started_at"]))
    lines = [f"{match['title']} — {stamp}", ""]
    if profile.key == "volleyball":
        set_state = sports.volleyball_sets(profile, events)
        lines.append(
            f"{labels['sets']}: A {set_state['wins'][0]} – {set_state['wins'][1]} B"
        )
        if set_state["sets"]:
            lines.append(
                "  " + ", ".join(f"{item['a']}–{item['b']}" for item in set_state["sets"])
            )
        if not set_state["complete"] and (set_state["current"][0] or set_state["current"][1]):
            lines.append(
                f"{labels['current_set']}: {set_state['number']} · "
                f"{set_state['current'][0]}–{set_state['current'][1]}"
            )
    else:
        lines.append(f"{labels['score']}: A {score[0]} – {score[1]} B")

    for items, label in (
        (scored, scoring_label),
        (errors, labels["errors"]),
        (assists, labels["assists"]),
    ):
        if not items:
            continue
        lines.append(label + ":")
        for event in sorted(items, key=lambda item: float(item.get("minute") or 0)):
            who = event.get("name") or "?"
            lines.append(
                f"  {int(float(event.get('minute') or 0))}′ {who} "
                f"({_team_letter(event.get('team'))})"
            )
    if not goals and not assists:
        lines.append(labels["none_volleyball"] if profile.key == "volleyball" else labels["none"])

    lines.append("")
    lines.append(labels["teams"] + ":")
    for letter, key in (("A", "0"), ("B", "1")):
        row = teams.get(key) or {}
        lines.append(
            f"  {letter}: {int(row.get('players') or 0)} {labels['players']} · "
            f"{float(row.get('distance_m') or 0):.1f} m · "
            f"{float(row.get('top_speed_kmh') or 0):.1f} km/s"
        )

    runners = sorted(rows, key=lambda row: -float(row.get("distance_m") or 0))[:3]
    if runners:
        lines.append("")
        lines.append(labels["runners"] + ":")
        for row in runners:
            who = names.get(int(row.get("tracker_id", -1))) or "?"
            lines.append(f"  {who} {float(row.get('distance_m') or 0):.1f} m")

    fastest = sorted(rows, key=lambda row: -float(row.get("top_speed_kmh") or 0))[:2]
    if fastest:
        line = ", ".join(
            f"{names.get(int(row.get('tracker_id', -1))) or '?'} "
            f"{float(row.get('top_speed_kmh') or 0):.1f} km/s"
            for row in fastest
        )
        lines.append(f"{labels['fastest']}: {line}")

    lines.append("")
    lines.append(labels["note"])
    return "\n".join(lines)


@app.get("/api/matches/{match_id}/report.txt")
def match_report_text(match_id: str, request: Request, lang: str = "tr") -> Response:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return Response(_report_text(match, lang), media_type="text/plain; charset=utf-8")


def _events_with_names(match_id: str) -> list[dict]:
    names = db.match_names(match_id)
    return [
        {**event, "name": names.get(event["tracker_id"])}
        for event in db.list_events(match_id)
    ]


def _sets_state(profile, events: list[dict]):
    """Voleybol set durumunu tek yerden üretir; diğer branşlarda None."""
    return sports.volleyball_sets(profile, events) if profile.key == "volleyball" else None


@app.get("/api/matches/{match_id}/events")
def get_match_events(match_id: str, request: Request) -> dict:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    events = _events_with_names(match_id)
    return {"events": events, "sets": _sets_state(sports.get(match.get("sport")), events)}


@app.post("/api/matches/{match_id}/events")
def add_match_event(match_id: str, payload: EventPayload, request: Request) -> dict:
    """Gol/asist kaydı ekler; dakika verilmezse canlı maçta oyun saatinden alınır."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    profile = sports.get(match.get("sport"))
    kind = payload.kind.strip().lower()
    if kind not in profile.event_kinds:
        raise HTTPException(status_code=400, detail="Geçersiz olay türü.")

    players = _match_players(match)
    row = None
    if payload.tracker_id is not None:
        row = next(
            (item for item in players if int(item.get("tracker_id", -1)) == payload.tracker_id),
            None,
        )
        if row is None:
            raise HTTPException(status_code=400, detail="Oyuncu bu maçta yok.")

    running = live.get(match_id)
    if payload.minute is not None:
        minute = max(0.0, float(payload.minute))
    elif running is not None and match["status"] in {"live", "connecting"}:
        minute = running.elapsed_active() / 60.0
    else:
        minute = 0.0

    team = int(row.get("team") or 0) if row else 0
    event = db.add_event(
        match_id,
        payload.tracker_id,
        team,
        kind,
        round(minute, 1),
    )
    event["name"] = db.match_names(match_id).get(event["tracker_id"])
    return {"event": event}


@app.delete("/api/matches/{match_id}/events/{event_id}")
def delete_match_event(match_id: str, event_id: int, request: Request) -> dict:
    group = current_group(request)
    if db.get_match(match_id, group["id"]) is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    if not db.delete_event(match_id, event_id):
        raise HTTPException(status_code=404, detail="Olay bulunamadı.")
    return {"ok": True}


def _recount_teams(rows: list[dict]) -> dict:
    """Oyuncu satırlarından takım özetlerini yeniden üretir."""
    teams = {}
    for team in (0, 1):
        members = [row for row in rows if int(row.get("team") or 0) == team]
        teams[str(team)] = {
            "players": len(members),
            "distance_m": round(sum(float(row.get("distance_m") or 0) for row in members), 1),
            "top_speed_kmh": max(
                (float(row.get("top_speed_kmh") or 0) for row in members), default=0.0
            ),
        }
    return teams


def _merge_rows(source: dict, target: dict) -> dict:
    """İki izi tek oyuncuda birleştirir: mesafe/sprint toplanır, tepe hız maksimumdur."""
    zones = {}
    for name in pipeline.ZONES:
        zones[name] = round(
            float((source.get("zones") or {}).get(name, 0))
            + float((target.get("zones") or {}).get(name, 0)),
            1,
        )
    source_samples = float(source.get("samples") or 0)
    target_samples = float(target.get("samples") or 0)
    samples = source_samples + target_samples
    average = 0.0
    if samples:
        average = (
            float(source.get("avg_speed_kmh") or 0) * source_samples
            + float(target.get("avg_speed_kmh") or 0) * target_samples
        ) / samples
    return {
        **target,
        "distance_m": round(
            float(source.get("distance_m") or 0) + float(target.get("distance_m") or 0), 1
        ),
        "top_speed_kmh": round(
            max(
                float(source.get("top_speed_kmh") or 0),
                float(target.get("top_speed_kmh") or 0),
            ),
            1,
        ),
        "avg_speed_kmh": round(average, 1),
        "samples": int(samples),
        "sprints": int(source.get("sprints") or 0) + int(target.get("sprints") or 0),
        "high_speed_m": round(
            float(source.get("high_speed_m") or 0)
            + float(target.get("high_speed_m") or 0),
            1,
        ),
        "zones": zones,
    }


def _remap_analysis(analysis: dict, source: int, target: int, mode: str) -> dict:
    """Analizdeki iki oyuncu kaydını takas eder ya da birleştirir."""
    rows = analysis.get("oyuncular") or []
    by_id = {int(row.get("tracker_id", -1)): row for row in rows}
    if source not in by_id or target not in by_id:
        raise HTTPException(status_code=400, detail="Oyuncu numarası bu maçta yok.")
    if source == target:
        raise HTTPException(status_code=400, detail="İki farklı oyuncu seç.")

    heat = analysis.setdefault("isi_haritasi", {}).setdefault("oyuncular", {})
    heat_source = heat.pop(str(source), None)
    heat_target = heat.pop(str(target), None)

    if mode == "swap":
        by_id[source]["tracker_id"], by_id[target]["tracker_id"] = target, source
        if heat_source:
            heat[str(target)] = heat_source
        if heat_target:
            heat[str(source)] = heat_target
        rows = sorted(rows, key=lambda row: -float(row.get("distance_m") or 0))
    else:
        merged = _merge_rows(by_id[source], by_id[target])
        rows = [
            merged if row is by_id[target] else row
            for row in rows
            if row is not by_id[source]
        ]
        rows.sort(key=lambda row: -float(row.get("distance_m") or 0))
        combined = dict(heat_target or {})
        for cell, count in (heat_source or {}).items():
            combined[str(cell)] = combined.get(str(cell), 0) + int(count)
        if combined:
            heat[str(target)] = combined

    analysis["oyuncular"] = rows
    analysis["takimlar"] = _recount_teams(rows)
    analysis["en_cok_kosan"] = rows[:3]
    analysis["en_hizli"] = sorted(
        rows, key=lambda row: -float(row.get("top_speed_kmh") or 0)
    )[:3]
    return analysis


@app.post("/api/matches/{match_id}/remap")
def remap_player(match_id: str, payload: RemapPayload, request: Request) -> dict:
    """Yanlış eşleşen oyuncu kimliklerini takas eder ya da iki kaydı birleştirir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    if payload.source == payload.target:
        raise HTTPException(status_code=400, detail="İki farklı oyuncu seç.")
    mode = (payload.mode or "swap").strip().lower()
    if mode not in {"swap", "merge"}:
        raise HTTPException(status_code=400, detail="Geçersiz işlem.")
    if live.get(match_id) is not None and match["status"] in {"live", "connecting"}:
        raise HTTPException(
            status_code=409, detail="Canlı maçta kimlik düzeltilemez; maç bitince dene."
        )

    path = _analysis_path(match)
    analysis = _analysis_of(match)

    _remap_analysis(analysis, payload.source, payload.target, mode)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    _ANALYSIS_CACHE.clear()

    # İsimler ve olaylar aynı kimlikleri izlesin.
    names = db.match_names(match_id)
    if mode == "swap":
        source_name = names.pop(payload.source, None)
        target_name = names.pop(payload.target, None)
        if target_name:
            names[payload.source] = target_name
        if source_name:
            names[payload.target] = source_name
    else:
        if not names.get(payload.target) and names.get(payload.source):
            names[payload.target] = names[payload.source]
        names.pop(payload.source, None)
    db.set_match_names(match_id, group["id"], names)
    db.remap_events(match_id, payload.source, payload.target, mode == "swap")

    # Kart, rapor ve ısı önbellekleri eski kimliklerle kalmasın.
    _invalidate_outputs(match_id)
    shutil.rmtree(heatmap.CACHE_DIR / match_id, ignore_errors=True)

    stats = dict(match.get("stats") or {})
    stats["players"] = analysis["oyuncular"]
    stats["teams"] = analysis["takimlar"]
    db.update_match(match_id, stats=stats)
    return {"players": analysis["oyuncular"], "teams": analysis["takimlar"]}


def _heat_buckets(match: dict) -> tuple[dict, dict, dict]:
    """Canlı maçta bellekten, bitmiş maçta analiz dosyasından ısı verisi."""
    running = live.get(match["id"])
    if running is not None and match["status"] in {"live", "connecting"}:
        return running.heat_snapshot()

    raw = _analysis_of(match).get("isi_haritasi") or {}

    def convert(source: dict) -> dict:
        return {
            int(key): {int(cell): int(count) for cell, count in (cells or {}).items()}
            for key, cells in (source or {}).items()
        }

    ball = raw.get("top") or {}
    return convert(raw.get("oyuncular")), convert(raw.get("takimlar")), {
        int(cell): int(count) for cell, count in ball.items()
    }


def _heat_scope(
    scope: str, players: dict, teams: dict, ball: Optional[dict] = None
) -> tuple[str, dict]:
    """İstenen kapsamı doğrular ve önbellek anahtarıyla birlikte döndürür."""
    value = (scope or "all").strip().lower()
    if value[:1] == "p" and value[1:].isdigit() and int(value[1:]) in players:
        tracker_id = int(value[1:])
        return value, players[tracker_id]
    if value in {"team0", "team1"}:
        return value, teams.get(int(value[-1]), {})
    if value == "ball":
        return value, ball or {}
    return "all", heatmap.merge(players.values()) or heatmap.merge(teams.values())


@app.get("/api/matches/{match_id}/heatmap.png")
def match_heatmap(
    match_id: str, request: Request, scope: str = "all", download: int = 0
) -> Response:
    """Saha ısı haritasını PNG olarak verir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    players, teams, ball = _heat_buckets(match)
    scope_key, bucket = _heat_scope(scope, players, teams, ball)
    if not bucket:
        raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")

    running = live.get(match_id)
    if running is not None and match["status"] in {"live", "connecting"}:
        data = heatmap.render(bucket, match.get("sport"))
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
        return Response(
            content=data, media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    safe = "".join(char for char in scope_key if char.isalnum()) or "all"
    path = heatmap.CACHE_DIR / match_id / f"{safe}.png"
    if not path.is_file():
        data = heatmap.render(bucket, match.get("sport"))
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"bakiappi-{match_id}-{safe}.png" if download else None,
        headers={"Cache-Control": "private, no-cache"},
    )


def _players_with_names(match: dict, analysis: dict) -> list:
    names = db.match_names(match["id"])
    return [
        {**row, "name": names.get(int(row.get("tracker_id", -1)))}
        for row in analysis.get("oyuncular") or []
    ]


def _heat_png_for(match: dict):
    """Takım raporuna gömülecek birleşik ısı haritası."""
    try:
        players_heat, teams_heat, _ = _heat_buckets(match)
    except HTTPException:
        return None
    bucket = heatmap.merge(players_heat.values())
    if not bucket:
        bucket = heatmap.merge(teams_heat.values())
    return heatmap.render(bucket, match.get("sport")) if bucket else None


@app.get("/api/matches/{match_id}/report.png")
def match_report(match_id: str, request: Request, lang: str = "fi") -> FileResponse:
    """Takım maç raporu (özet + ısı haritası + en çok koşanlar)."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    analysis = _analysis_of(match)
    profile = sports.get(match.get("sport"))

    language = "en" if lang.lower().startswith("en") else "fi"
    path = REPORTS_DIR / f"{match_id}-{language}.png"
    if not path.is_file():
        events = _events_with_names(match_id)
        score_text = ""
        if profile.key == "volleyball":
            state = sports.volleyball_sets(profile, events)
            sets_word, set_word = ("Erät", "Erä") if language == "fi" else ("Sets", "Set")
            if state["complete"]:
                score_text = f"{sets_word} {state['wins'][0]}–{state['wins'][1]}"
            else:
                score_text = (
                    f"{set_word} {state['number']} · {state['current'][0]}–{state['current'][1]}"
                    f" · {sets_word} {state['wins'][0]}–{state['wins'][1]}"
                )
        meta = cards.build_meta(
            title=match.get("title") or analysis.get("mac") or "",
            date=analysis.get("tarih") or "",
            group=group["name"],
            match_id=match_id,
            elapsed_s=analysis.get("sure_sn") or 0,
            sport=match.get("sport"),
            score_text=score_text,
        )
        data = cards.render_match_report(
            meta,
            _players_with_names(match, analysis),
            analysis.get("takimlar") or {},
            _heat_png_for(match),
            language,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"bakiappi-{match_id}-raportti.png",
        headers={"Cache-Control": "private, no-cache"},
    )


@app.get("/api/matches/{match_id}/export.csv")
def export_csv(match_id: str, request: Request) -> Response:
    """Oyuncu istatistiklerini Excel uyumlu CSV olarak indirir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    analysis = _analysis_of(match)

    profile = sports.get(match.get("sport"))
    scoring_label = "points" if profile.key in {"basketball", "volleyball"} else "goals"
    goals: dict = {}
    assists: dict = {}
    for event in db.list_events(match_id):
        tracker_id = event.get("tracker_id")
        if tracker_id is None:
            continue
        if event["kind"] == "assist":
            assists[int(tracker_id)] = assists.get(int(tracker_id), 0) + 1
        elif event["kind"] in profile.event_points and event["kind"] not in profile.error_kinds:
            goals[int(tracker_id)] = goals.get(int(tracker_id), 0) + profile.event_points.get(
                event["kind"], 1
            )

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        [
            "sport", "player", "team", "distance_m", "top_speed_kmh", "avg_speed_kmh",
            "samples", "sprints", "walk_m", "jog_m", "run_m", "sprint_m",
            scoring_label, "assists",
        ]
    )
    for row in _players_with_names(match, analysis):
        zones = row.get("zones") or {}
        tracker_id = int(row.get("tracker_id", -1))
        writer.writerow(
            [
                profile.key,
                row.get("name") or f"#{row.get('tracker_id')}",
                "A" if int(row.get("team") or 0) == 0 else "B",
                row.get("distance_m", 0),
                row.get("top_speed_kmh", 0),
                row.get("avg_speed_kmh", 0),
                row.get("samples", 0),
                row.get("sprints", 0),
                zones.get("walk", 0),
                zones.get("jog", 0),
                zones.get("run", 0),
                zones.get("sprint", 0),
                goals.get(tracker_id, 0),
                assists.get(tracker_id, 0),
            ]
        )

    data = "\ufeff" + buffer.getvalue()
    return Response(
        content=data.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="bakiappi-{match_id}.csv"'},
    )


@app.post("/api/matches/{match_id}/share")
def create_share(match_id: str, request: Request) -> dict:
    """Herkese açık (salt okunur) maç bağlantısı üretir."""
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    token = match.get("share_token") or secrets.token_urlsafe(16)
    db.set_share_token(match_id, token)
    return {"token": token, "path": f"/s/{token}"}


@app.delete("/api/matches/{match_id}/share")
def revoke_share(match_id: str, request: Request) -> dict:
    group = current_group(request)
    if db.get_match(match_id, group["id"]) is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    db.set_share_token(match_id, None)
    return {"ok": True}


def _public_match(token: str) -> dict:
    match = db.match_by_token(token)
    if match is None or match["status"] not in {"finished", "live", "connecting"}:
        raise HTTPException(status_code=404, detail="Bağlantı geçersiz.")
    return match


@app.get("/api/public/{token}")
def public_match(token: str) -> dict:
    """Herkese açık maç özeti; canlı maçta bellekten taze skor ve dakika döner."""
    match = _public_match(token)
    running = live.get(match["id"])
    is_live = running is not None and match["status"] in {"live", "connecting"}
    names = db.match_names(match["id"])

    if is_live:
        stats = running.snapshot()
        players = [
            {**row, "name": names.get(int(row.get("tracker_id", -1)))}
            for row in stats.get("players") or []
        ]
        payload = {
            "sport": match.get("sport") or "football",
            "period_name": (match.get("stats") or {}).get("period_name"),
            "date": stats.get("started_at_text") or "",
            "elapsed_s": stats.get("elapsed_s") or 0,
            "teams": stats.get("teams") or {},
            "paused": bool(stats.get("paused")),
            "period": stats.get("period"),
            "stage": stats.get("stage") or "",
        }
    else:
        analysis = _analysis_of(match)
        players = _players_with_names(match, analysis)
        payload = {
            "sport": match.get("sport") or "football",
            "period_name": analysis.get("period_name"),
            "date": analysis.get("tarih") or "",
            "elapsed_s": analysis.get("sure_sn") or 0,
            "teams": analysis.get("takimlar") or {},
            "paused": False,
            "period": None,
            "stage": "",
        }

    events = _events_with_names(match["id"])
    return {
        "title": match["title"],
        "group": match.get("group_name") or "",
        "status": "live" if is_live else match["status"],
        **payload,
        "players": players,
        "events": events,
        "sets": _sets_state(sports.get(match.get("sport")), events),
    }


@app.get("/api/public/{token}/heatmap.png")
def public_heatmap(token: str, scope: str = "all") -> Response:
    match = _public_match(token)
    players, teams, ball = _heat_buckets(match)
    _, bucket = _heat_scope(scope, players, teams, ball)
    if not bucket:
        raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
    data = heatmap.render(bucket, match.get("sport"))
    if data is None:
        raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
    return Response(
        content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=60"}
    )


@app.get("/s/{token}")
def share_page(token: str) -> FileResponse:
    return FileResponse(
        BASE / "static" / "share.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/roster")
def roster(request: Request) -> dict:
    """İsim atanmış oyuncuların maçlar arası toplamları (sezon görünümü)."""
    group = current_group(request)
    totals: dict = {}
    selected = group.get("sport") or sports.DEFAULT

    for match in db.list_matches(group["id"], sport=selected):
        analysis = _analysis_of(match, missing_ok=True)
        if analysis is None:
            continue

        names = db.match_names(match["id"])
        player_ids = db.match_player_ids(match["id"])
        for row in analysis.get("oyuncular") or []:
            tracker_id = int(row.get("tracker_id", -1))
            name = (names.get(tracker_id) or "").strip()
            if not name:
                continue
            entry = totals.setdefault(
                player_ids.get(tracker_id) or name.casefold(),
                {
                    "player_id": player_ids.get(tracker_id),
                    "name": name,
                    "matches": 0,
                    "distance_m": 0.0,
                    "top_speed_kmh": 0.0,
                    "sprints": 0,
                },
            )
            entry["matches"] += 1
            entry["distance_m"] += float(row.get("distance_m") or 0.0)
            entry["top_speed_kmh"] = max(
                entry["top_speed_kmh"], float(row.get("top_speed_kmh") or 0.0)
            )
            entry["sprints"] += int(row.get("sprints") or 0)

    players = sorted(totals.values(), key=lambda row: -row["distance_m"])
    for row in players:
        row["distance_m"] = round(row["distance_m"], 1)
        row["top_speed_kmh"] = round(row["top_speed_kmh"], 1)
    return {
        "players": players,
        "names": [row["name"] for row in players],
        "squad": db.list_players(group["id"]),
    }


def _trend_payload(group_id: int, limit: int = 12, sport: Optional[str] = None) -> dict:
    """İsimli oyuncuların maç bazlı gelişimini (mesafe/hız/sprint) hazırlar."""
    limit = max(2, min(int(limit), 30))
    matches = [
        match
        for match in db.list_matches(group_id, sport=sport)
        if match["status"] == "finished" and match.get("analysis_path")
    ]
    matches.sort(key=lambda match: match["started_at"])
    matches = matches[-limit:]

    series: dict = {}
    for match in matches:
        analysis = _analysis_of(match, missing_ok=True)
        if analysis is None:
            continue
        names = db.match_names(match["id"])
        player_ids = db.match_player_ids(match["id"])
        for row in analysis.get("oyuncular") or []:
            tracker_id = int(row.get("tracker_id", -1))
            player_id = player_ids.get(tracker_id)
            if not player_id:
                continue
            entry = series.setdefault(
                player_id,
                {
                    "player_id": player_id,
                    "name": names.get(tracker_id) or "?",
                    "points": {},
                },
            )
            entry["points"][match["id"]] = {
                "distance_m": round(float(row.get("distance_m") or 0), 1),
                "top_speed_kmh": round(float(row.get("top_speed_kmh") or 0), 1),
                "sprints": int(row.get("sprints") or 0),
            }

    payload = []
    for entry in series.values():
        points = [
            {
                "match_id": match["id"],
                **entry["points"].get(match["id"], {}),
            }
            for match in matches
        ]
        payload.append(
            {"player_id": entry["player_id"], "name": entry["name"], "points": points}
        )
    payload.sort(
        key=lambda item: -sum(
            float(point.get("distance_m") or 0) for point in item["points"]
        )
    )
    return {
        "matches": [
            {
                "id": match["id"],
                "title": match["title"],
                "started_at": match["started_at"],
            }
            for match in matches
        ],
        "series": payload,
    }


@app.get("/api/roster/trends")
def roster_trends(request: Request, limit: int = 12) -> dict:
    group = current_group(request)
    return _trend_payload(group["id"], limit, group.get("sport") or sports.DEFAULT)


@app.get("/api/public/{token}/trends")
def public_trends(token: str) -> dict:
    """Paylaşılan maçtaki isimli oyuncuların gelişimi (yalnızca onların maçları)."""
    match = _public_match(token)
    payload = _trend_payload(match["group_id"], 12, match.get("sport"))
    allowed = set(db.match_player_ids(match["id"]).values())
    series = [row for row in payload["series"] if row["player_id"] in allowed]
    used = {match["id"]} | {
        point["match_id"]
        for row in series
        for point in row["points"]
        if point.get("distance_m") is not None
    }
    kept = [entry for entry in payload["matches"] if entry["id"] in used]
    kept_ids = {entry["id"] for entry in kept}
    return {
        "matches": kept,
        "series": [
            {**row, "points": [p for p in row["points"] if p["match_id"] in kept_ids]}
            for row in series
        ],
    }


@app.get("/api/matches/{match_id}/analysis/file")
def download_analysis(match_id: str, request: Request) -> FileResponse:
    group = current_group(request)
    match = db.get_match(match_id, group["id"])
    if match is None:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    path = _analysis_path(match)
    return FileResponse(path, media_type="application/json", filename=path.name)


# ------------------------------------------------------------- video analizi
def save_job(job: dict) -> None:
    (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job), encoding="utf-8")


def load_jobs() -> None:
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(job, dict) and "id" in job:
            JOBS[job["id"]] = job


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()
        if job.get("status") in {"done", "error"}:
            save_job(job)


def make_browser_playable(path: Path) -> bool:
    """OpenCV mp4v yazar; Chromium oynatamaz. Videoyu H.264'e çevirir."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.is_file():
        return False
    converted = path.with_name(path.stem + "-h264.mp4")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(converted),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not converted.is_file():
        return False
    converted.replace(path)
    return True


def run_job(
    job_id: str,
    source: Path,
    target: Path,
    max_seconds: float,
    sport: str = "football",
    calibration: Optional[dict] = None,
) -> None:
    def on_progress(payload: dict) -> None:
        update_job(job_id, status="running", **payload)

    try:
        stats = pipeline.analyze(
            source_path=str(source),
            target_path=str(target),
            max_seconds=max_seconds,
            progress=on_progress,
            sport=sport,
            calibration=calibration,
        )
        update_job(job_id, stage="video hazırlanıyor", percent=99.5)
        stats["browser_playable"] = make_browser_playable(target)
        update_job(
            job_id,
            status="done",
            stage="tamamlandı",
            percent=100.0,
            stats=stats,
            finished_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        update_job(
            job_id,
            status="error",
            stage="hata",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Geliştirme sırasında tarayıcı eski arayüz dosyalarını kullanmasın."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/samples/{name}")
def sample_video(name: str) -> FileResponse:
    """Stüdyoda kare önizlemesi/kalibrasyon için örnek videoyu verir."""
    path = (pipeline.DATA_DIR / Path(name).name).resolve()
    if not path.is_file() or path.parent != pipeline.DATA_DIR.resolve():
        raise HTTPException(status_code=404, detail="Örnek video bulunamadı.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/samples")
def list_samples() -> list:
    return [
        {"name": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
        for path in sorted(pipeline.DATA_DIR.glob("*.mp4"))
        if "-radar" not in path.name
    ]


@app.post("/api/analyze")
async def start_analysis(
    sample: Optional[str] = Form(default=None),
    max_seconds: float = Form(default=10.0),
    sport: str = Form(default="football"),
    calibration: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
) -> dict:
    job_id = uuid.uuid4().hex[:12]
    profile = sports.get(sport)
    parsed_calibration = None
    if calibration:
        try:
            parsed_calibration = json.loads(calibration)
            sports.calibration_matrix(
                profile,
                parsed_calibration.get("points"),
                parsed_calibration.get("extent", "full"),
            )
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Kalibrasyon geçersiz: {exc}") from exc

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
        source = UPLOADS_DIR / f"{job_id}{suffix}"
        with source.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        label = file.filename
    elif sample:
        source = (pipeline.DATA_DIR / Path(sample).name).resolve()
        if not source.is_file() or source.parent != pipeline.DATA_DIR.resolve():
            raise HTTPException(status_code=404, detail="Örnek video bulunamadı.")
        label = source.name
    else:
        raise HTTPException(status_code=400, detail="Video seçilmedi.")

    with _lock:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "sırada",
            "percent": 0.0,
            "frames_done": 0,
            "frames_total": 0,
            "source": label,
            "sport": profile.key,
            "calibration": parsed_calibration,
            "error": None,
            "stats": None,
            "created_at": time.time(),
        }

    threading.Thread(
        target=run_job,
        args=(
            job_id,
            source,
            JOBS_DIR / f"{job_id}.mp4",
            float(max_seconds),
            profile.key,
            parsed_calibration,
        ),
        daemon=True,
    ).start()
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs() -> list:
    with _lock:
        jobs = sorted(JOBS.values(), key=lambda job: job.get("created_at", 0), reverse=True)
    return [
        {"id": job["id"], "status": job["status"], "source": job["source"]}
        for job in jobs
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="İş bulunamadı.")
    return job


@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    video = JOBS_DIR / f"{job_id}.mp4"
    if not video.is_file():
        raise HTTPException(status_code=404, detail="Video hazır değil.")
    return FileResponse(video, media_type="video/mp4", filename=f"bakiappi-{job_id}.mp4")


@app.get("/studio")
def studio() -> RedirectResponse:
    return RedirectResponse("/studio.html")


load_jobs()

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
