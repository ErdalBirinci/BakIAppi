"""BakIAppi — maç sonu oyuncu kartı (1080×1350 PNG, Pillow).

Kart, analiz dosyasındaki oyuncu satırlarından üretilir; tarayıcı ya da harici
servis gerekmez. Aynı maç için rozet sahipleri bir kez hesaplanır.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app import sports

W, H = 1080, 1350
PAD = 64
FOOTER_Y = 1218
HEAT_THUMB = (220, 128)
HEAT_RADIUS = 14
COMPASS = (50, 30)

INK = (231, 238, 247)
MUTED = (143, 163, 184)
PINK = (255, 20, 147)
BLUE = (0, 191, 255)
ORANGE = (255, 170, 90)
GOLD = (255, 215, 0)
GOLD_DIM = (201, 185, 120)
LINE = (30, 43, 58)
PANEL = (11, 17, 24)

BG_TOP = (19, 29, 44)
BG_BOTTOM = (7, 10, 15)

MAX_SPEED_KMH = 30.0

_BOLD = (
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_REGULAR = (
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

# Rozetler: anahtar -> (Fince ad, İngilizce karşılık)
BADGES = {
    "loyly": ("Löyly-Legenda", "Sauna Legend"),
    "pikakiitaja": ("Pikakiitäjä", "Lightning Glider"),
    "kiuas": ("Kiuas-Kuningas", "Sauna Stove King"),
    "sisu": ("Sisu-Sissi", "Grit Ranger"),
    "avantouimari": ("Avantouimari", "Ice Swimmer"),
    "salmiakki": ("Salmiakki-Soturi", "Salty Licorice Warrior"),
}

RANKS = (
    (500.0, "Jänis", "Hare"),
    (1000.0, "Poro", "Reindeer"),
    (1500.0, "Hirvi", "Moose"),
    (2000.0, "Karhu", "Bear"),
    (float("inf"), "Kontio", "Great Bear"),
)

AVANTOUIMARI_M = 2000.0

TEXT = {
    "fi": {
        "team": "JOUKKUE {team}",
        "rank": "ARVO",
        "distance": "MATKA",
        "top": "HUIPPUNOPEUS",
        "avg": "KESKINOPEUS",
        "speed_title": "NOPEUS",
        "speed_top": "Huippu",
        "speed_avg": "Keski",
        "player": "Pelaaja",
        "team_distance": "JOUKKUEEN MATKA",
        "match_top": "OTTELUN KOVIN",
        "heat": "LÄMPÖKARTTA",
        "samples": "{samples} näytettä · {duration}",
    },
    "en": {
        "team": "TEAM {team}",
        "rank": "RANK",
        "distance": "DISTANCE",
        "top": "TOP SPEED",
        "avg": "AVERAGE SPEED",
        "speed_title": "SPEED",
        "speed_top": "Top",
        "speed_avg": "Average",
        "player": "Player",
        "team_distance": "TEAM DISTANCE",
        "match_top": "FASTEST SPRINT",
        "heat": "HEATMAP",
        "samples": "{samples} samples · {duration}",
    },
}


@lru_cache(maxsize=96)
def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    for candidate in (_BOLD if bold else _REGULAR):
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _gradient(width: int, height: int, top: tuple, bottom: tuple) -> Image.Image:
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    column = top_arr * (1.0 - ramp) + bottom_arr * ramp
    array = np.repeat(column, width, axis=1).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _background() -> Image.Image:
    image = _gradient(W, H, BG_TOP, BG_BOTTOM).convert("RGBA")

    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    for x in range(0, W, 120):
        draw.line([(x, 0), (x, H)], fill=(255, 255, 255, 9))
    for y in range(0, H, 120):
        draw.line([(0, y), (W, y)], fill=(255, 255, 255, 9))
    image = Image.alpha_composite(image, grid)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((700, -320, 1340, 320), fill=(*PINK, 96))
    glow = glow.filter(ImageFilter.GaussianBlur(150))
    return Image.alpha_composite(image, glow)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    size: int,
    color: tuple,
    bold: bool = True,
    anchor: str = "la",
) -> None:
    draw.text(xy, text, font=_font(size, bold), fill=color, anchor=anchor)


def _fit(
    draw: ImageDraw.ImageDraw, text: str, size: int, max_width: float, bold: bool = True
) -> ImageFont.ImageFont:
    """Metin max_width'e sığana kadar bir kademe küçültür."""
    while size > 22 and draw.textlength(text, font=_font(size, bold)) > max_width:
        size -= 4
    return _font(size, bold)


def _distance(metres: float) -> tuple[str, str]:
    """Mesafeyi kısa maçlarda metre, uzun maçlarda kilometre olarak yazar."""
    metres = max(0.0, float(metres or 0.0))
    return (f"{metres / 1000:.1f}", "km") if metres >= 1000 else (f"{metres:.0f}", "m")


def _pill_width(
    draw: ImageDraw.ImageDraw, text: str, size: int, pad_x: int = 20, suffix: str = ""
) -> float:
    width = draw.textlength(text, font=_font(size, True)) + pad_x * 2
    if suffix:
        width += draw.textlength(suffix, font=_font(size - 8, True)) + 10
    return width


def _pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    size: int,
    fg: tuple,
    bg: tuple,
    border: tuple,
    pad_x: int = 20,
    pad_y: int = 10,
    suffix: str = "",
    suffix_color: tuple = GOLD_DIM,
) -> float:
    """Çerçeveli etiket çizer ve sağ kenarını döndürür."""
    font = _font(size, True)
    width = _pill_width(draw, text, size, pad_x, suffix)
    height = size + pad_y * 2
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=height / 2, fill=bg, outline=border, width=2
    )
    draw.text((x + pad_x, y + height / 2), text, font=font, fill=fg, anchor="lm")
    if suffix:
        small = _font(size - 8, True)
        draw.text(
            (x + pad_x + draw.textlength(text, font=font) + 10, y + height / 2),
            suffix,
            font=small,
            fill=suffix_color,
            anchor="lm",
        )
    return x + width


def match_awards(players: Iterable[dict]) -> dict[int, list[str]]:
    """Bir maçtaki rozet sahiplerini hesaplar (oyuncu kimliği -> rozet anahtarları)."""
    rows = [row for row in players if row.get("tracker_id") is not None]
    awards: dict[int, list[str]] = {}

    def give(player: Optional[dict], key: str) -> None:
        if player is None:
            return
        bucket = awards.setdefault(int(player["tracker_id"]), [])
        if key not in bucket:
            bucket.append(key)

    if rows:
        give(max(rows, key=lambda row: row.get("distance_m", 0.0)), "loyly")
        give(max(rows, key=lambda row: row.get("top_speed_kmh", 0.0)), "pikakiitaja")
        give(max(rows, key=lambda row: row.get("avg_speed_kmh", 0.0)), "kiuas")
        give(max(rows, key=lambda row: row.get("samples", 0)), "sisu")
    for row in rows:
        if row.get("distance_m", 0.0) >= AVANTOUIMARI_M:
            give(row, "avantouimari")
    # Katılım rozeti en sona eklenir; üst sıradaki rozetler önceliklidir.
    for row in rows:
        give(row, "salmiakki")
    return awards


def rank_of(distance_m: float) -> tuple[str, str]:
    for limit, fi, en in RANKS:
        if distance_m < limit:
            return fi, en
    return RANKS[-1][1], RANKS[-1][2]


def _speed_panel(
    draw: ImageDraw.ImageDraw,
    max_speed: float,
    x: float,
    y: float,
    width: float,
    height: float,
    player: dict,
    labels: dict,
    duration_text: str,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=26, fill=(*PANEL, 235), outline=LINE, width=2
    )
    _text(draw, (x + 30, y + 24), labels["speed_title"], 19, MUTED)

    rows = (
        (labels["speed_top"], float(player.get("top_speed_kmh") or 0), BLUE, y + 70),
        (labels["speed_avg"], float(player.get("avg_speed_kmh") or 0), PINK, y + 150),
    )
    unit_font = _font(18, False)
    unit_width = draw.textlength("km/h", font=unit_font)
    for label, value, color, row_y in rows:
        _text(draw, (x + 30, row_y), label, 22, INK, bold=False)
        draw.text(
            (x + width - 30, row_y + 6), "km/h", font=unit_font, fill=MUTED, anchor="ra"
        )
        draw.text(
            (x + width - 30 - unit_width - 8, row_y),
            f"{value:.1f}",
            font=_font(30),
            fill=color,
            anchor="ra",
        )

        track_y = row_y + 52
        track_left, track_right = x + 30, x + width - 30
        draw.rounded_rectangle(
            (track_left, track_y, track_right, track_y + 12), radius=6, fill=(26, 36, 50)
        )
        ratio = max(0.0, min(1.0, value / max(1.0, max_speed)))
        fill_right = track_left + (track_right - track_left) * ratio
        if fill_right - track_left > 6:
            draw.rounded_rectangle(
                (track_left, track_y, fill_right, track_y + 12), radius=6, fill=color
            )
        draw.ellipse(
            (fill_right - 7, track_y - 6, fill_right + 7, track_y + 18),
            fill=(255, 255, 255),
        )

    _text(
        draw,
        (x + 30, y + height - 40),
        labels["samples"].format(
            samples=player.get("samples", 0), duration=duration_text
        ),
        18,
        MUTED,
        bold=False,
    )


@lru_cache(maxsize=1)
def _heat_mask() -> Image.Image:
    """Küçük resmin yuvarlatılmış köşe maskesi (boyut sabit, bir kez üretilir)."""
    width, height = HEAT_THUMB
    mask = Image.new("L", HEAT_THUMB, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=HEAT_RADIUS, fill=255
    )
    return mask


def _scale(value: float, start: float, end: float) -> float:
    return start + value * (end - start)


def _draw_pitch_compass(
    draw: ImageDraw.ImageDraw,
    box: Optional[tuple[float, float, float, float]],
    x: float,
    y: float,
) -> None:
    """Kırpılan pencerenin sahadaki yerini gösteren küçük pusula."""
    width, height = COMPASS
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=7, fill=(10, 16, 22, 242), outline=LINE
    )
    left, top = x + 4, y + 4
    right, bottom = x + width - 4, y + height - 4
    draw.rectangle((left, top, right, bottom), outline=(70, 92, 116))
    middle = (left + right) / 2
    draw.line((middle, top, middle, bottom), fill=(48, 66, 88))
    if not box:
        return
    bx0 = _scale(box[0], left, right)
    by0 = _scale(box[1], top, bottom)
    draw.rectangle(
        (
            bx0,
            by0,
            max(_scale(box[2], left, right), bx0 + 2),
            max(_scale(box[3], top, bottom), by0 + 2),
        ),
        outline=GOLD,
        width=2,
    )


def _draw_heat_preview(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    heat_png: bytes,
    box: Optional[tuple[float, float, float, float]],
    label: str,
) -> None:
    """Sağ alt köşeye kırpılmış ısı haritası ve saha pusulası çizer."""
    width, height = HEAT_THUMB
    x, y = W - PAD - width, FOOTER_Y - 14 - height
    _text(draw, (x, y - 30), label, 16, MUTED)

    thumb = Image.open(io.BytesIO(heat_png)).convert("RGB")
    if box:
        thumb = thumb.crop(
            tuple(
                int(value * size)
                for value, size in zip(box, (thumb.width, thumb.height) * 2)
            )
        )
    thumb = thumb.resize(HEAT_THUMB, Image.Resampling.LANCZOS)
    image.paste(thumb, (x, y), _heat_mask())
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=HEAT_RADIUS, outline=LINE, width=2
    )
    _draw_pitch_compass(draw, box, x + width - COMPASS[0] - 6, y + height - COMPASS[1] - 6)


def render_player_card(
    player: dict,
    meta: dict,
    awards: Optional[list[str]] = None,
    lang: str = "fi",
    heat_png: Optional[bytes] = None,
    heat_box: Optional[tuple[float, float, float, float]] = None,
) -> bytes:
    """Tek oyuncunun 1080×1350 kartını PNG baytları olarak üretir."""
    lang = normalize_lang(lang)
    labels = TEXT[lang]
    awards = [key for key in (awards or []) if key in BADGES][:3]

    image = _background()
    draw = ImageDraw.Draw(image)

    # --- üst bar ---------------------------------------------------------
    _text(draw, (PAD, PAD - 8), "BakI", 46, INK)
    brand_width = draw.textlength("BakI", font=_font(46))
    _text(draw, (PAD + brand_width, PAD - 8), "Appi", 46, PINK)

    title = str(meta.get("title") or "")
    draw.text((W - PAD, PAD - 2), title, font=_fit(draw, title, 28, 520), fill=INK, anchor="ra")
    _text(draw, (W - PAD, PAD + 44), str(meta.get("date") or ""), 23, MUTED, bold=False, anchor="ra")
    _text(
        draw,
        (W - PAD, PAD + 78),
        _sport_label(meta, lang),
        18,
        _sport_color(meta),
        anchor="ra",
    )

    # --- kimlik ----------------------------------------------------------
    team = int(player.get("team") or 0)
    team_color = PINK if team == 0 else BLUE
    shirt = str(player.get("shirt") or player.get("tracker_id") or "")
    center = (PAD + 76, 300)
    draw.ellipse(
        (center[0] - 76, center[1] - 76, center[0] + 76, center[1] + 76),
        fill=(27, 15, 24) if team == 0 else (9, 26, 36),
        outline=team_color,
        width=4,
    )
    draw.text(center, shirt, font=_fit(draw, shirt, 66, 120), fill=INK, anchor="mm")

    name = str(player.get("name") or f"{labels['player']} #{player.get('tracker_id')}")
    name_x = PAD + 180
    chip_right = _pill(
        draw,
        name_x,
        226,
        labels["team"].format(team="A" if team == 0 else "B"),
        20,
        team_color,
        (27, 15, 24) if team == 0 else (9, 26, 36),
        team_color,
        pad_x=16,
        pad_y=8,
    )
    draw.text((name_x, 268), name, font=_fit(draw, name, 76, W - PAD - name_x), fill=INK)

    rank_fi, rank_en = rank_of(float(player.get("distance_m") or 0.0))
    _pill(
        draw,
        chip_right + 10,
        226,
        labels["rank"],
        18,
        GOLD,
        (26, 21, 5),
        (107, 87, 20),
        pad_x=14,
        pad_y=8,
        suffix=rank_fi if lang == "fi" else rank_en,
    )

    # --- ana istatistikler ----------------------------------------------
    distance_value, distance_unit = _distance(player.get("distance_m"))
    boxes = (
        (PAD, 380, 390, distance_value, distance_unit, labels["distance"], INK, 88),
        (472, 380, 262, f"{float(player.get('top_speed_kmh') or 0):.1f}", "km/h", labels["top"], BLUE, 60),
        (752, 380, 264, f"{float(player.get('avg_speed_kmh') or 0):.1f}", "km/h", labels["avg"], INK, 60),
    )
    for x, y, width, value, unit, label, color, value_size in boxes:
        draw.rounded_rectangle(
            (x, y, x + width, y + 190), radius=26, fill=(*PANEL, 235), outline=LINE, width=2
        )
        font = _font(value_size)
        draw.text((x + 28, y + 30), value, font=font, fill=color)
        value_width = draw.textlength(value, font=font)
        _text(draw, (x + 28 + value_width + 8, y + 30 + value_size - 30), unit, 22, MUTED, bold=False)
        _text(draw, (x + 28, y + 148), label, 18, MUTED)

    # --- hız paneli ------------------------------------------------------
    _speed_panel(
        draw,
        float(meta.get("max_speed_kmh") or MAX_SPEED_KMH),
        64,
        596,
        952,
        280,
        player,
        labels,
        str(meta.get("duration") or "—"),
    )

    # --- rozetler --------------------------------------------------------
    if awards:
        x, y = 64, 906
        for key in awards:
            fi, en = BADGES[key]
            badge_width = _pill_width(draw, fi, 26, 22, en)
            if x > 64 and x + badge_width > W - PAD:
                x, y = 64, y + 74
            x = _pill(
                draw,
                x,
                y,
                fi,
                26,
                GOLD,
                (26, 21, 5),
                (107, 87, 20),
                pad_x=22,
                pad_y=14,
                suffix=en,
            ) + 12

    # --- takım özeti -----------------------------------------------------
    team_distance = meta.get("team_distance")
    if team_distance is not None:
        team_value, team_unit = _distance(team_distance)
        _text(draw, (PAD, 1082), labels["team_distance"], 16, MUTED)
        _text(draw, (PAD, 1108), f"{team_value} {team_unit}", 32, INK)
        _text(draw, (PAD + 470, 1082), labels["match_top"], 16, MUTED)
        _text(
            draw,
            (PAD + 470, 1108),
            f"{float(meta.get('match_top') or 0):.1f} km/h",
            32,
            BLUE,
        )

    # --- ısı haritası önizlemesi ----------------------------------------
    if heat_png:
        _draw_heat_preview(image, draw, heat_png, heat_box, labels["heat"])

    # --- alt bilgi -------------------------------------------------------
    draw.line([(PAD, FOOTER_Y), (W - PAD, FOOTER_Y)], fill=LINE, width=2)
    group = str(meta.get("group") or "")
    _text(draw, (PAD, 1246), f"{group} · BakIAppi.app", 22, MUTED, bold=False)
    _text(draw, (W - PAD, 1246), str(meta.get("code") or ""), 22, MUTED, bold=False, anchor="ra")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# ------------------------------------------------------------------ dosyalar
CARD_ROOT = Path(__file__).resolve().parent / "data" / "cards"
CARD_VERSION = 2


def normalize_lang(lang: str) -> str:
    """Kartta yalnızca Fince ve İngilizce var; diğer diller Finceye düşer."""
    return "en" if str(lang).lower().startswith("en") else "fi"


def card_path(match_id: str, tracker_id: int, lang: str) -> Path:
    return CARD_ROOT / match_id / f"{tracker_id}-{normalize_lang(lang)}-v{CARD_VERSION}.png"


def prune_stale() -> None:
    """Eski şema/sürüm kartlarını siler; sürüm değişince bayat görsel kalmasın."""
    if not CARD_ROOT.is_dir():
        return
    suffix = f"-v{CARD_VERSION}"
    for path in CARD_ROOT.rglob("*.png"):
        if not path.stem.endswith(suffix):
            path.unlink(missing_ok=True)


def build_meta(
    *,
    title: str,
    date: str,
    group: str,
    match_id: str,
    elapsed_s: float,
    teams: Optional[dict] = None,
    team: int = 0,
    sport: Optional[str] = None,
    score_text: str = "",
) -> dict:
    """Kartın sabit alanları + oyuncunun takım özeti (+ branş hız ölçeği)."""
    seconds = max(0, int(round(float(elapsed_s or 0))))
    stats = (teams or {}).get(str(team), {}) or {}
    profile = sports.get(sport)
    return {
        "title": title or "",
        "sport": profile.key,
        "score_text": score_text or "",
        "max_speed_kmh": profile.max_speed_kmh,
        "date": date or "",
        "group": group or "",
        "code": f"KKS-{match_id[:4].upper()}",
        "duration": f"{seconds // 60}:{seconds % 60:02d}",
        "team_distance": stats.get("distance_m"),
        "match_top": stats.get("top_speed_kmh"),
    }


def invalidate_match(match_id: str) -> None:
    """Maçın önbelleklenmiş kartlarını ve boş kart klasörünü siler."""
    folder = CARD_ROOT / match_id
    if folder.is_dir():
        for path in folder.glob("*.png"):
            path.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass


SPORT_TEXT = {
    "fi": {"football": "JALKAPALLO", "basketball": "KORIPALLO", "volleyball": "LENTOPALLO"},
    "en": {"football": "FOOTBALL", "basketball": "BASKETBALL", "volleyball": "VOLLEYBALL"},
}


def _sport_label(meta: dict, lang: str) -> str:
    key = str(meta.get("sport") or "football")
    return SPORT_TEXT.get(normalize_lang(lang), SPORT_TEXT["fi"]).get(key, key.upper())


def _sport_color(meta: dict) -> tuple:
    key = str(meta.get("sport") or "football")
    if key == "basketball":
        return BLUE
    if key == "volleyball":
        return ORANGE
    return MUTED


REPORT_TEXT = {
    "fi": {
        "players": "PELAAJAT",
        "team": "JOUKKUE {team}",
        "top": "KOVIN VAUHTI",
        "top_players": "ENTEN KULKENEET",
        "sprints": "sprinttiä",
        "no_heat": "Ei lämpökarttadataa tälle ottelulle",
    },
    "en": {
        "players": "PLAYERS",
        "team": "TEAM {team}",
        "top": "FASTEST SPRINT",
        "top_players": "TOP DISTANCE",
        "sprints": "sprints",
        "no_heat": "No heatmap data for this match",
    },
}


def render_match_report(
    meta: dict,
    players: list[dict],
    teams: dict,
    heat_png: Optional[bytes] = None,
    lang: str = "fi",
) -> bytes:
    """Takım maç raporu: özet, ısı haritası ve en çok koşanlar (1080×1350)."""
    lang = normalize_lang(lang)
    labels = REPORT_TEXT[lang]
    image = _background()
    draw = ImageDraw.Draw(image)

    _text(draw, (PAD, PAD - 8), "BakI", 46, INK)
    brand_width = draw.textlength("BakI", font=_font(46))
    _text(draw, (PAD + brand_width, PAD - 8), "Appi", 46, PINK)

    title = str(meta.get("title") or "")
    draw.text((W - PAD, PAD - 2), title, font=_fit(draw, title, 28, 520), fill=INK, anchor="ra")
    _text(draw, (W - PAD, PAD + 44), str(meta.get("date") or ""), 23, MUTED, bold=False, anchor="ra")
    _text(
        draw,
        (W - PAD, PAD + 78),
        _sport_label(meta, lang),
        18,
        _sport_color(meta),
        anchor="ra",
    )
    if meta.get("score_text"):
        _text(
            draw,
            (W - PAD, PAD + 96),
            str(meta["score_text"]),
            20,
            GOLD,
            anchor="ra",
        )

    team_a = (teams or {}).get("0") or {}
    team_b = (teams or {}).get("1") or {}
    fastest = max(
        float(team_a.get("top_speed_kmh") or 0.0), float(team_b.get("top_speed_kmh") or 0.0)
    )
    distance_a = _distance(team_a.get("distance_m"))
    distance_b = _distance(team_b.get("distance_m"))
    boxes = (
        (str(len(players)), "", labels["players"], INK, 72),
        (distance_a[0], distance_a[1], labels["team"].format(team="A"), PINK, 56),
        (distance_b[0], distance_b[1], labels["team"].format(team="B"), BLUE, 56),
        (f"{fastest:.1f}", "km/h", labels["top"], BLUE, 56),
    )
    box_w, gap = 220, 24
    for index, (value, unit, label, color, size) in enumerate(boxes):
        x = PAD + index * (box_w + gap)
        draw.rounded_rectangle(
            (x, 190, x + box_w, 360), radius=24, fill=(*PANEL, 235), outline=LINE, width=2
        )
        font = _font(size)
        draw.text((x + 24, 212), value, font=font, fill=color)
        value_width = draw.textlength(value, font=font)
        if unit:
            _text(
                draw,
                (x + 24 + value_width + 8, 212 + size - 26),
                unit,
                20,
                MUTED,
                bold=False,
            )
        _text(draw, (x + 24, 318), label, 16, MUTED)

    if heat_png:
        heat = Image.open(io.BytesIO(heat_png)).convert("RGB").resize(
            (952, 555), Image.Resampling.LANCZOS
        )
        image.paste(heat, (PAD, 400))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((PAD, 400, PAD + 952, 955), radius=20, outline=LINE, width=2)
    else:
        draw.rounded_rectangle(
            (PAD, 400, PAD + 952, 955), radius=20, fill=(*PANEL, 160), outline=LINE, width=2
        )
        _text(
            draw,
            (W / 2, 660),
            labels["no_heat"],
            26,
            MUTED,
            bold=False,
            anchor="mm",
        )

    _text(draw, (PAD, 994), labels["top_players"], 16, MUTED)
    ordered = sorted(players, key=lambda row: -float(row.get("distance_m") or 0.0))[:3]
    for index, row in enumerate(ordered):
        y = 1030 + index * 56
        name = str(row.get("name") or f"{TEXT[lang]['player']} #{row.get('tracker_id')}")
        _text(draw, (PAD, y), f"{index + 1}.", 26, GOLD)
        draw.text((PAD + 56, y), name, font=_fit(draw, name, 30, 600), fill=INK)
        value, unit = _distance(row.get("distance_m"))
        _text(draw, (W - PAD, y + 2), f"{value} {unit}", 26, MUTED, bold=False, anchor="ra")
        _text(
            draw,
            (W - PAD, y + 32),
            f"{int(row.get('sprints') or 0)} {labels['sprints']}",
            16,
            MUTED,
            bold=False,
            anchor="ra",
        )

    draw.line([(PAD, 1218), (W - PAD, 1218)], fill=LINE, width=2)
    _text(draw, (PAD, 1246), f"{meta.get('group') or ''} · BakIAppi.app", 22, MUTED, bold=False)
    _text(draw, (W - PAD, 1246), str(meta.get("code") or ""), 22, MUTED, bold=False, anchor="ra")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def save_card(
    match_id: str,
    tracker_id: int,
    lang: str,
    player: dict,
    meta: dict,
    awards: Optional[list[str]] = None,
    heat_png: Optional[bytes] = None,
    heat_box: Optional[tuple[float, float, float, float]] = None,
) -> Path:
    """Kartı diske yazar (varsa yeniden üretmez) ve yolunu döndürür."""
    path = card_path(match_id, tracker_id, lang)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            render_player_card(
                player,
                meta,
                awards,
                normalize_lang(lang),
                heat_png=heat_png,
                heat_box=heat_box,
            )
        )
    return path
