import io
from pathlib import Path

path = Path("app/server.py")
raw = io.open(path, "r", encoding="utf-8", newline="").read()
text = raw.replace("\r\n", "\n")


def sub(old, new, count=1):
    global text
    assert text.count(old) == count, (text.count(old), old[:80])
    text = text.replace(old, new)


sub(
    "import csv\nimport io\nimport json\nimport secrets\n",
    "import csv\nimport io\nimport json\nimport secrets\n",
)
sub(
    "from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile\n",
    "import cv2\nfrom fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile\n",
)
sub(
    "from app import cards, db, heatmap, live, pipeline\n",
    "from app import cards, db, heatmap, live, pipeline, sports\n",
)
sub(
    'UPLOADS_DIR = BASE / "data" / "uploads"\n',
    'UPLOADS_DIR = BASE / "data" / "uploads"\nFRAMES_DIR = BASE / "data" / "frames"\n',
)
sub(
    'UPLOADS_DIR.mkdir(parents=True, exist_ok=True)\n',
    'UPLOADS_DIR.mkdir(parents=True, exist_ok=True)\nFRAMES_DIR.mkdir(parents=True, exist_ok=True)\n',
)

sub(
    '''    # Yarıda kalan maçları hata saymak yerine kaldığı yerden devam ettirmeyi dene.
    for match in db.live_matches(started_before=SERVER_STARTED_AT):
        live.start(match["id"], match["group_id"], match["title"], match["source_url"])''',
    '''    # Yarıda kalan maçları hata saymak yerine kaldığı yerden devam ettirmeyi dene.
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
        )''',
)

sub(
    '''class MatchPayload(BaseModel):
    title: str = ""
    source_url: str
''',
    '''class MatchPayload(BaseModel):
    title: str = ""
    source_url: str
    sport: Optional[str] = None
    calibration: Optional[dict] = None


class SportPayload(BaseModel):
    sport: str


class CalibrationPayload(BaseModel):
    points: list
    extent: str = "full"
''',
)

sub(
    '''@app.get("/api/auth/me")
def me(request: Request) -> dict:
    return current_group(request)
''',
    '''@app.get("/api/auth/me")
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
''',
)

sub(
    '''def _owned_video(match_id: str, request: Request) -> Path:''',
    '''def _capture_frame(match: dict, position_s: float = 0.0) -> Path:
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


def _owned_video(match_id: str, request: Request) -> Path:''',
)

sub(
    '''    title = payload.title.strip() or f"Maç {time.strftime('%d.%m.%Y %H:%M')}"
    match_id = uuid.uuid4().hex[:12]
    db.create_match(match_id, group["id"], title, source)
    live.start(match_id, group["id"], title, source)
    return {"id": match_id, "title": title}''',
    '''    if payload.sport is not None and not sports.is_valid(payload.sport):
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

    default_title = "Karşılaşma" if profile.key == "basketball" else "Maç"
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
    return {"id": match_id, "title": title, "sport": profile.key}''',
)

sub(
    '''    live.forget_finished()
    db.update_match(match_id, status="live", error=None, finished_at=None)
    live.start(match_id, group["id"], match["title"], match["source_url"])
    return {"ok": True}''',
    '''    live.forget_finished()
    db.update_match(match_id, status="live", error=None, finished_at=None)
    live.start(
        match_id,
        group["id"],
        match["title"],
        match["source_url"],
        match.get("sport") or "football",
        match.get("calibration"),
    )
    return {"ok": True}''',
)

sub(
    '''@app.get("/api/matches/{match_id}/video")
def match_video(match_id: str, request: Request) -> FileResponse:''',
    '''@app.get("/api/matches/{match_id}/frame.jpg")
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
    return {"ok": True, "calibration": calibration}


@app.get("/api/matches/{match_id}/video")
def match_video(match_id: str, request: Request) -> FileResponse:''',
)

sub(
    '''def _heat_buckets(match: dict) -> tuple[dict, dict]:
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

    return convert(raw.get("oyuncular")), convert(raw.get("takimlar"))''',
    '''def _heat_buckets(match: dict) -> tuple[dict, dict, dict]:
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
    }''',
)

sub(
    '''def _heat_scope(scope: str, players: dict, teams: dict) -> tuple[str, dict]:
    """İstenen kapsamı doğrular ve önbellek anahtarıyla birlikte döndürür."""
    value = (scope or "all").strip().lower()
    if value[:1] == "p" and value[1:].isdigit() and int(value[1:]) in players:
        tracker_id = int(value[1:])
        return value, players[tracker_id]
    if value in {"team0", "team1"}:
        return value, teams.get(int(value[-1]), {})
    return "all", heatmap.merge(players.values()) or heatmap.merge(teams.values())''',
    '''def _heat_scope(
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
    return "all", heatmap.merge(players.values()) or heatmap.merge(teams.values())''',
)

sub(
    '''    players, teams = _heat_buckets(match)
    scope_key, bucket = _heat_scope(scope, players, teams)''',
    '''    players, teams, ball = _heat_buckets(match)
    scope_key, bucket = _heat_scope(scope, players, teams, ball)''',
)
sub(
    '''        data = heatmap.render(bucket)
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
        return Response(''',
    '''        data = heatmap.render(bucket, match.get("sport"))
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
        return Response(''',
)
sub(
    '''    if not path.is_file():
        data = heatmap.render(bucket)
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")''',
    '''    if not path.is_file():
        data = heatmap.render(bucket, match.get("sport"))
        if data is None:
            raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")''',
)

sub(
    '''def _heat_png_for(match: dict):
    """Takım raporuna gömülecek birleşik ısı haritası."""
    try:
        players_heat, teams_heat = _heat_buckets(match)
    except HTTPException:
        return None
    bucket = heatmap.merge(players_heat.values())
    if not bucket:
        bucket = heatmap.merge(teams_heat.values())
    return heatmap.render(bucket) if bucket else None''',
    '''def _heat_png_for(match: dict):
    """Takım raporuna gömülecek birleşik ısı haritası."""
    try:
        players_heat, teams_heat, _ = _heat_buckets(match)
    except HTTPException:
        return None
    bucket = heatmap.merge(players_heat.values())
    if not bucket:
        bucket = heatmap.merge(teams_heat.values())
    return heatmap.render(bucket, match.get("sport")) if bucket else None''',
)

sub(
    '''        try:
            player_heat, _ = _heat_buckets(match)
        except HTTPException:
            player_heat = {}''',
    '''        try:
            player_heat, _, _ = _heat_buckets(match)
        except HTTPException:
            player_heat = {}''',
)
sub(
    '''            heat_png=heatmap.render(cells),
            heat_box=heatmap.focus_box(cells),''',
    '''            heat_png=heatmap.render(cells, match.get("sport")),
            heat_box=heatmap.focus_box(cells, sport=match.get("sport")),''',
)

sub(
    '''            teams=analysis.get("takimlar"),
            team=int(player.get("team") or 0),
        )''',
    '''            teams=analysis.get("takimlar"),
            team=int(player.get("team") or 0),
            sport=match.get("sport"),
        )''',
)
sub(
    '''            group=group["name"],
            match_id=match_id,
            elapsed_s=analysis.get("sure_sn") or 0,
        )
        data = cards.render_match_report(''',
    '''            group=group["name"],
            match_id=match_id,
            elapsed_s=analysis.get("sure_sn") or 0,
            sport=match.get("sport"),
        )
        data = cards.render_match_report(''',
)

sub(
    '''    data = heatmap.render(bucket)
    if data is None:
        raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
    return Response(
        content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=60"}
    )''',
    '''    data = heatmap.render(bucket, match.get("sport"))
    if data is None:
        raise HTTPException(status_code=404, detail="Isı haritası verisi yok.")
    return Response(
        content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=60"}
    )''',
)
sub(
    '''    match = _public_match(token)
    players, teams = _heat_buckets(match)
    _, bucket = _heat_scope(scope, players, teams)''',
    '''    match = _public_match(token)
    players, teams, ball = _heat_buckets(match)
    _, bucket = _heat_scope(scope, players, teams, ball)''',
)

sub(
    '''    goals = [event for event in events if event.get("kind") == "goal"]
    assists = [event for event in events if event.get("kind") == "assist"]
    score = {0: 0, 1: 0}
    for event in goals:
        try:
            score[int(event.get("team") or 0)] += 1
        except (KeyError, TypeError, ValueError):
            pass''',
    '''    profile = sports.get(match.get("sport"))
    goals = [event for event in events if event.get("kind") in profile.event_points]
    assists = [event for event in events if event.get("kind") == "assist"]
    score = {0: 0, 1: 0}
    for event in goals:
        try:
            score[int(event.get("team") or 0)] += profile.event_points.get(
                event.get("kind"), 1
            )
        except (KeyError, TypeError, ValueError):
            pass
    scoring_label = labels["baskets"] if profile.key == "basketball" else labels["goals"]''',
)
sub(
    '''    for items, label in ((goals, labels["goals"]), (assists, labels["assists"])):''',
    '''    for items, label in ((goals, scoring_label), (assists, labels["assists"])):''',
)
sub(
    '''        "goals": "Goller",
        "assists": "Asistler",''',
    '''        "goals": "Goller",
        "baskets": "Basketler",
        "assists": "Asistler",''',
)
sub(
    '''        "goals": "Maalit",
        "assists": "Syötöt",''',
    '''        "goals": "Maalit",
        "baskets": "Korit",
        "assists": "Syötöt",''',
)
sub(
    '''        "goals": "Goals",
        "assists": "Assists",''',
    '''        "goals": "Goals",
        "baskets": "Baskets",
        "assists": "Assists",''',
)

sub(
    '''    kind = payload.kind.strip().lower()
    if kind not in db.EVENT_KINDS:
        raise HTTPException(status_code=400, detail="Geçersiz olay türü.")''',
    '''    profile = sports.get(match.get("sport"))
    kind = payload.kind.strip().lower()
    if kind not in profile.event_kinds:
        raise HTTPException(status_code=400, detail="Geçersiz olay türü.")''',
)

sub(
    '''    goals: dict = {}
    assists: dict = {}
    for event in db.list_events(match_id):
        tracker_id = event.get("tracker_id")
        if tracker_id is None:
            continue
        bucket = goals if event["kind"] == "goal" else assists
        bucket[int(tracker_id)] = bucket.get(int(tracker_id), 0) + 1

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        [
            "player", "team", "distance_m", "top_speed_kmh", "avg_speed_kmh",
            "samples", "sprints", "walk_m", "jog_m", "run_m", "sprint_m",
            "goals", "assists",
        ]
    )''',
    '''    profile = sports.get(match.get("sport"))
    scoring_label = "points" if profile.key == "basketball" else "goals"
    goals: dict = {}
    assists: dict = {}
    for event in db.list_events(match_id):
        tracker_id = event.get("tracker_id")
        if tracker_id is None:
            continue
        if event["kind"] == "assist":
            assists[int(tracker_id)] = assists.get(int(tracker_id), 0) + 1
        elif event["kind"] in profile.event_points:
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
    )''',
)
sub(
    '''        writer.writerow(
            [
                row.get("name") or f"#{row.get('tracker_id')}",''',
    '''        writer.writerow(
            [
                profile.key,
                row.get("name") or f"#{row.get('tracker_id')}",''',
)

sub(
    '''        payload = {
            "date": stats.get("started_at_text") or "",''',
    '''        payload = {
            "sport": match.get("sport") or "football",
            "period_name": (match.get("stats") or {}).get("period_name"),
            "date": stats.get("started_at_text") or "",''',
)
sub(
    '''        payload = {
            "date": analysis.get("tarih") or "",''',
    '''        payload = {
            "sport": match.get("sport") or "football",
            "period_name": analysis.get("period_name"),
            "date": analysis.get("tarih") or "",''',
)

sub(
    '''    group = current_group(request)
    totals: dict = {}

    for match in db.list_matches(group["id"]):
        analysis = _analysis_of(match, missing_ok=True)''',
    '''    group = current_group(request)
    totals: dict = {}
    selected = group.get("sport") or sports.DEFAULT

    for match in db.list_matches(group["id"], sport=selected):
        analysis = _analysis_of(match, missing_ok=True)''',
)

sub(
    '''def _trend_payload(group_id: int, limit: int = 12) -> dict:
    """İsimli oyuncuların maç bazlı gelişimini (mesafe/hız/sprint) hazırlar."""
    limit = max(2, min(int(limit), 30))
    matches = [
        match
        for match in db.list_matches(group_id)
        if match["status"] == "finished" and match.get("analysis_path")
    ]''',
    '''def _trend_payload(group_id: int, limit: int = 12, sport: Optional[str] = None) -> dict:
    """İsimli oyuncuların maç bazlı gelişimini (mesafe/hız/sprint) hazırlar."""
    limit = max(2, min(int(limit), 30))
    matches = [
        match
        for match in db.list_matches(group_id, sport=sport)
        if match["status"] == "finished" and match.get("analysis_path")
    ]''',
)
sub(
    '''def roster_trends(request: Request, limit: int = 12) -> dict:
    group = current_group(request)
    return _trend_payload(group["id"], limit)''',
    '''def roster_trends(request: Request, limit: int = 12) -> dict:
    group = current_group(request)
    return _trend_payload(group["id"], limit, group.get("sport") or sports.DEFAULT)''',
)
sub(
    '''    match = _public_match(token)
    payload = _trend_payload(match["group_id"], 12)''',
    '''    match = _public_match(token)
    payload = _trend_payload(match["group_id"], 12, match.get("sport"))''',
)

sub(
    '''def run_job(job_id: str, source: Path, target: Path, max_seconds: float) -> None:''',
    '''def run_job(
    job_id: str,
    source: Path,
    target: Path,
    max_seconds: float,
    sport: str = "football",
    calibration: Optional[dict] = None,
) -> None:''',
)
sub(
    '''        stats = pipeline.analyze(
            source_path=str(source),
            target_path=str(target),
            max_seconds=max_seconds,
            progress=on_progress,
        )''',
    '''        stats = pipeline.analyze(
            source_path=str(source),
            target_path=str(target),
            max_seconds=max_seconds,
            progress=on_progress,
            sport=sport,
            calibration=calibration,
        )''',
)

sub(
    '''@app.post("/api/analyze")
async def start_analysis(
    sample: Optional[str] = Form(default=None),
    max_seconds: float = Form(default=10.0),
    file: Optional[UploadFile] = File(default=None),
) -> dict:
    job_id = uuid.uuid4().hex[:12]
''',
    '''@app.post("/api/analyze")
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
''',
)

sub(
    '''            "source": label,
            "error": None,
            "stats": None,
            "created_at": time.time(),
        }

    threading.Thread(
        target=run_job,
        args=(job_id, source, JOBS_DIR / f"{job_id}.mp4", float(max_seconds)),
        daemon=True,
    ).start()''',
    '''            "source": label,
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
    ).start()''',
)

sub(
    '''@app.get("/api/samples")
def list_samples() -> list:''',
    '''@app.get("/api/samples/{name}")
def sample_video(name: str) -> FileResponse:
    """Stüdyoda kare önizlemesi/kalibrasyon için örnek videoyu verir."""
    path = (pipeline.DATA_DIR / Path(name).name).resolve()
    if not path.is_file() or path.parent != pipeline.DATA_DIR.resolve():
        raise HTTPException(status_code=404, detail="Örnek video bulunamadı.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/samples")
def list_samples() -> list:''',
)

io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace("\n", "\r\n"))
print("server patched")
