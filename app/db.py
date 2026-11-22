"""SQLite: halısaha grupları, oturumlar ve maç kayıtları."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DB_PATH = DATA_DIR / "kikkusisu.db"  # eski ad: mevcut veri dosyasıyla aynı kalsın
ANALYSIS_DIR = DATA_DIR / "analysis"
VIDEO_DIR = DATA_DIR / "videos"

SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    sport TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    group_id INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    group_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    stats_json TEXT,
    analysis_path TEXT,
    sport TEXT NOT NULL DEFAULT 'football',
    calibration TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_group ON matches(group_id, started_at DESC);
CREATE TABLE IF NOT EXISTS match_players (
    match_id TEXT NOT NULL,
    tracker_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (match_id, tracker_id)
);
CREATE TABLE IF NOT EXISTS match_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    tracker_id INTEGER,
    team INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL,
    minute REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_match ON match_events(match_id, minute);
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (group_id, name_key)
);
CREATE INDEX IF NOT EXISTS idx_players_group ON players(group_id, name_key);
"""

PBKDF2_ROUNDS = 200_000
MAX_NAME_LENGTH = 40
EVENT_KINDS = (
    "goal",
    "assist",
    "basket",
    "basket3",
    "ace",
    "kill",
    "block",
    "error",
    "dig",
    "set",
    "reception",
)
SESSION_TTL = 30 * 24 * 3600


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.executescript(SCHEMA)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(matches)")}
        if "share_token" not in columns:
            connection.execute("ALTER TABLE matches ADD COLUMN share_token TEXT")
        if "sport" not in columns:
            connection.execute(
                "ALTER TABLE matches ADD COLUMN sport TEXT NOT NULL DEFAULT 'football'"
            )
        if "calibration" not in columns:
            connection.execute("ALTER TABLE matches ADD COLUMN calibration TEXT")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_share ON matches(share_token)"
        )

        group_columns = {row["name"] for row in connection.execute("PRAGMA table_info(groups)")}
        if "sport" not in group_columns:
            connection.execute("ALTER TABLE groups ADD COLUMN sport TEXT")

        # Kadro kimliği: eski maç isimleri de oyuncu kaydına bağlanır.
        player_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(match_players)")
        }
        if "player_id" not in player_columns:
            connection.execute("ALTER TABLE match_players ADD COLUMN player_id INTEGER")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_match_players_player ON match_players(player_id)"
        )
        for row in connection.execute(
            "SELECT match_players.match_id AS match_id, match_players.tracker_id AS tracker_id,"
            " match_players.name AS name, matches.group_id AS group_id"
            " FROM match_players JOIN matches ON matches.id = match_players.match_id"
            " WHERE match_players.player_id IS NULL"
        ).fetchall():
            player_id = _upsert_player(connection, row["group_id"], row["name"])
            connection.execute(
                "UPDATE match_players SET player_id = ? WHERE match_id = ? AND tracker_id = ?",
                (player_id, row["match_id"], row["tracker_id"]),
            )


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ROUNDS
    ).hex()


def create_group(name: str, username: str, password: str) -> int:
    """Yeni grup hesabı açar; kullanıcı adı alınmışsa ValueError yükseltir."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    try:
        with connect() as connection:
            cursor = connection.execute(
                "INSERT INTO groups (name, username, password_hash, salt, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (name.strip(), username.strip(), password_hash, salt, time.time()),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("Bu kullanıcı adı zaten alınmış.") from exc


def verify_login(username: str, password: str) -> Optional[dict]:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM groups WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row is None:
        return None
    if not secrets.compare_digest(_hash_password(password, row["salt"]), row["password_hash"]):
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "username": row["username"],
        "sport": row["sport"],
    }


def group_name(group_id: int) -> str:
    with connect() as connection:
        row = connection.execute(
            "SELECT name FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
    return row["name"] if row else ""


def create_session(group_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with connect() as connection:
        connection.execute(
            "INSERT INTO sessions (token, group_id, created_at) VALUES (?, ?, ?)",
            (token, group_id, time.time()),
        )
    return token


def group_for_token(token: Optional[str]) -> Optional[dict]:
    """Oturum jetonunu doğrular; süresi geçmiş jetonları siler."""
    if not token:
        return None
    with connect() as connection:
        row = connection.execute(
            "SELECT groups.id, groups.name, groups.username, groups.sport, sessions.created_at"
            " FROM sessions JOIN groups ON groups.id = sessions.group_id"
            " WHERE sessions.token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["created_at"] + SESSION_TTL < time.time():
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
    return {
        "id": row["id"],
        "name": row["name"],
        "username": row["username"],
        "sport": row["sport"],
    }


def set_group_sport(group_id: int, sport: str) -> None:
    """Grubun varsayılan branşını kaydeder."""
    with connect() as connection:
        connection.execute("UPDATE groups SET sport = ? WHERE id = ?", (sport, group_id))


def prune_sessions() -> int:
    """Süresi geçmiş oturumları siler; silinen sayısını döner."""
    cutoff = time.time() - SESSION_TTL
    with connect() as connection:
        cursor = connection.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
    return cursor.rowcount or 0


def delete_session(token: str) -> None:
    with connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


def create_match(
    match_id: str,
    group_id: int,
    title: str,
    source_url: str,
    sport: str = "football",
    calibration: Optional[str] = None,
) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO matches"
            " (id, group_id, title, source_url, status, started_at, sport, calibration)"
            " VALUES (?, ?, ?, ?, 'live', ?, ?, ?)",
            (match_id, group_id, title, source_url, time.time(), sport, calibration),
        )


def update_match(match_id: str, **fields) -> None:
    if not fields:
        return
    if "stats" in fields:
        fields["stats_json"] = json.dumps(fields.pop("stats"), ensure_ascii=False)
    columns = ", ".join(f"{name} = ?" for name in fields)
    with connect() as connection:
        connection.execute(
            f"UPDATE matches SET {columns} WHERE id = ?", (*fields.values(), match_id)
        )


def _match_row(row: sqlite3.Row) -> dict:
    stats = None
    if row["stats_json"]:
        try:
            stats = json.loads(row["stats_json"])
        except json.JSONDecodeError:
            stats = None
    calibration = None
    raw_calibration = row["calibration"] if "calibration" in row.keys() else None
    if raw_calibration:
        try:
            calibration = json.loads(raw_calibration)
        except json.JSONDecodeError:
            calibration = None
    return {
        "sport": row["sport"] or "football",
        "calibration": calibration,
        "id": row["id"],
        "group_id": row["group_id"],
        "title": row["title"],
        "source_url": row["source_url"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "analysis_path": row["analysis_path"],
        "share_token": row["share_token"],
        "stats": stats,
    }


def list_matches(group_id: int, sport: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM matches WHERE group_id = ?"
    parameters: list = [group_id]
    if sport:
        query += " AND sport = ?"
        parameters.append(sport)
    query += " ORDER BY started_at DESC LIMIT 100"
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_match_row(row) for row in rows]


def get_match(match_id: str, group_id: int) -> Optional[dict]:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM matches WHERE id = ? AND group_id = ?", (match_id, group_id)
        ).fetchone()
    return None if row is None else _match_row(row)


def set_share_token(match_id: str, token: Optional[str]) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE matches SET share_token = ? WHERE id = ?", (token, match_id)
        )


def ensure_share_token(match_id: str) -> str:
    """Maçın herkese açık jetonunu döner; yoksa üretip kaydeder."""
    with connect() as connection:
        row = connection.execute(
            "SELECT share_token FROM matches WHERE id = ?", (match_id,)
        ).fetchone()
        if row is not None and row["share_token"]:
            return row["share_token"]
        token = secrets.token_urlsafe(16)
        connection.execute(
            "UPDATE matches SET share_token = ? WHERE id = ?", (token, match_id)
        )
    return token


def match_by_token(token: str) -> Optional[dict]:
    """Herkese açık paylaşım bağlantısı için maç + grup adı."""
    if not token:
        return None
    with connect() as connection:
        row = connection.execute(
            "SELECT matches.*, groups.name AS group_name FROM matches"
            " JOIN groups ON groups.id = matches.group_id WHERE matches.share_token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    match = _match_row(row)
    match["group_name"] = row["group_name"]
    return match


def match_names(match_id: str) -> dict[int, str]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT tracker_id, name FROM match_players WHERE match_id = ?", (match_id,)
        ).fetchall()
    return {int(row["tracker_id"]): row["name"] for row in rows}


def norm_name(value: str) -> str:
    """İsim karşılaştırma anahtarı: boşluklar sadeleşir, büyük/küçük harf ayrımı kalkar."""
    return " ".join(str(value).split()).casefold()


def _upsert_player(connection: sqlite3.Connection, group_id: int, name: str) -> int:
    """Oyuncuyu kadroya ekler (varsa mevcut kimliği döner)."""
    cleaned = str(name).strip()[:MAX_NAME_LENGTH]
    connection.execute(
        "INSERT INTO players (group_id, name, name_key, created_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (group_id, name_key) DO NOTHING",
        (group_id, cleaned, norm_name(cleaned), time.time()),
    )
    row = connection.execute(
        "SELECT id FROM players WHERE group_id = ? AND name_key = ?",
        (group_id, norm_name(cleaned)),
    ).fetchone()
    return int(row["id"])


def set_match_names(match_id: str, group_id: int, names: dict[int, str]) -> None:
    """Maçın isim atamalarını tümüyle değiştirir; isimleri kadroya bağlar."""
    with connect() as connection:
        connection.execute("DELETE FROM match_players WHERE match_id = ?", (match_id,))
        rows = []
        for tracker_id, name in names.items():
            cleaned = str(name).strip()[:MAX_NAME_LENGTH]
            if not cleaned:
                continue
            player_id = _upsert_player(connection, group_id, cleaned)
            rows.append((match_id, int(tracker_id), cleaned, player_id))
        connection.executemany(
            "INSERT INTO match_players (match_id, tracker_id, name, player_id)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )


def match_player_ids(match_id: str) -> dict[int, int]:
    """Maçtaki tracker_id → kadro oyuncu kimliği eşlemesi."""
    with connect() as connection:
        rows = connection.execute(
            "SELECT tracker_id, player_id FROM match_players"
            " WHERE match_id = ? AND player_id IS NOT NULL",
            (match_id,),
        ).fetchall()
    return {int(row["tracker_id"]): int(row["player_id"]) for row in rows}


def list_players(group_id: int) -> list[dict]:
    """Grubun kadrosu: oyuncu, oynadığı maç sayısı ve son maç tarihi."""
    with connect() as connection:
        rows = connection.execute(
            "SELECT players.id, players.name,"
            " COUNT(DISTINCT match_players.match_id) AS matches,"
            " MAX(matches.started_at) AS last_match"
            " FROM players"
            " LEFT JOIN match_players ON match_players.player_id = players.id"
            " LEFT JOIN matches ON matches.id = match_players.match_id"
            " WHERE players.group_id = ?"
            " GROUP BY players.id, players.name"
            " ORDER BY players.name COLLATE NOCASE",
            (group_id,),
        ).fetchall()
    return [
        {"id": row["id"], "name": row["name"], "matches": row["matches"], "last_match": row["last_match"]}
        for row in rows
    ]


def rename_player(group_id: int, player_id: int, name: str) -> dict:
    """Kadro oyuncusunu yeniden adlandırır; aynı isim varsa kayıtları birleştirir."""
    cleaned = str(name).strip()[:MAX_NAME_LENGTH]
    if not cleaned:
        raise ValueError("İsim boş olamaz.")
    new_key = norm_name(cleaned)
    with connect() as connection:
        row = connection.execute(
            "SELECT id FROM players WHERE id = ? AND group_id = ?", (player_id, group_id)
        ).fetchone()
        if row is None:
            raise ValueError("Oyuncu bulunamadı.")

        existing = connection.execute(
            "SELECT id FROM players WHERE group_id = ? AND name_key = ?",
            (group_id, new_key),
        ).fetchone()
        if existing is not None and int(existing["id"]) != player_id:
            # Aynı isimli kayıt varsa iki oyuncu birleşir; kimlik korunur.
            target_id = int(existing["id"])
            connection.execute(
                "UPDATE match_players SET player_id = ? WHERE player_id = ?",
                (target_id, player_id),
            )
            connection.execute("DELETE FROM players WHERE id = ?", (player_id,))
        else:
            target_id = player_id
            connection.execute(
                "UPDATE players SET name = ?, name_key = ? WHERE id = ?",
                (cleaned, new_key, player_id),
            )
        connection.execute(
            "UPDATE match_players SET name = ? WHERE player_id = ?", (cleaned, target_id)
        )
    return {"id": target_id, "name": cleaned}


def delete_match(match_id: str, group_id: int) -> bool:
    """Maçı olayları ve isim atamalarıyla birlikte siler."""
    with connect() as connection:
        row = connection.execute(
            "SELECT id FROM matches WHERE id = ? AND group_id = ?", (match_id, group_id)
        ).fetchone()
        if row is None:
            return False
        connection.execute("DELETE FROM match_events WHERE match_id = ?", (match_id,))
        connection.execute("DELETE FROM match_players WHERE match_id = ?", (match_id,))
        connection.execute("DELETE FROM matches WHERE id = ?", (match_id,))
    return True


def _event_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "match_id": row["match_id"],
        "tracker_id": row["tracker_id"],
        "team": row["team"],
        "kind": row["kind"],
        "minute": row["minute"],
        "created_at": row["created_at"],
    }


def add_event(
    match_id: str, tracker_id: Optional[int], team: int, kind: str, minute: float
) -> dict:
    """Gol/asist kaydı ekler."""
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO match_events (match_id, tracker_id, team, kind, minute, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (match_id, tracker_id, int(team), kind, float(minute), time.time()),
        )
        row = connection.execute(
            "SELECT * FROM match_events WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return _event_row(row)


def list_events(match_id: str) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM match_events WHERE match_id = ? ORDER BY minute, id", (match_id,)
        ).fetchall()
    return [_event_row(row) for row in rows]


def delete_event(match_id: str, event_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM match_events WHERE id = ? AND match_id = ?", (event_id, match_id)
        )
    return cursor.rowcount > 0


def remap_events(match_id: str, source: int, target: int, swap: bool) -> None:
    """Kimlik düzeltmesinde olayları da taşır."""
    with connect() as connection:
        if swap:
            connection.execute(
                "UPDATE match_events SET tracker_id = -1 WHERE match_id = ? AND tracker_id = ?",
                (match_id, source),
            )
            connection.execute(
                "UPDATE match_events SET tracker_id = ? WHERE match_id = ? AND tracker_id = ?",
                (source, match_id, target),
            )
            connection.execute(
                "UPDATE match_events SET tracker_id = ? WHERE match_id = ? AND tracker_id = -1",
                (target, match_id),
            )
        else:
            connection.execute(
                "UPDATE match_events SET tracker_id = ? WHERE match_id = ? AND tracker_id = ?",
                (target, match_id, source),
            )


def live_matches(started_before: Optional[float] = None) -> list[dict]:
    """Yarıda kalan canlı maçlar; yeniden başlatmada devam etmek için gerekli alanlar.

    `started_before` verilirse yalnızca bu andan önce başlamış maçlar döner; böylece
    portu tutamayıp ölen ikinci bir süreç, yeni başlatılan maçı yanlışlıkla kesmez.
    """
    query = (
        "SELECT id, group_id, title, source_url, sport, calibration FROM matches"
        " WHERE status IN ('live', 'connecting')"
    )
    parameters: tuple = ()
    if started_before is not None:
        query += " AND started_at < ?"
        parameters = (started_before,)
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


init()
