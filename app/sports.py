"""Branş profilleri: futbol, basketbol ve voleybol için saha, hız bölgeleri ve olaylar.

Futbol hattı saha anahtar noktalarından otomatik homografi kurar. Basketbol
sahaları için anahtar nokta modeli bulunmadığından homografi, kullanıcının
görüntüde işaretlediği dört köşeden (tam ya da yarım saha) kurulur. Saha
ölçüleri, ısı haritası ızgarası, hız bölgeleri ve olay türleri burada tutulur;
analiz, canlı yayın, kartlar ve raporlar aynı profili kullanır.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "examples" / "soccer" / "data"

DEFAULT = "football"
PIXELS_PER_M = 10
PAD_PX = 12

COURT_FILL = (16, 32, 24)
COURT_LINE = (78, 122, 96)


@dataclass(frozen=True)
class SportProfile:
    key: str
    name: str
    name_en: str
    name_fi: str
    emoji: str
    court_length_cm: int
    court_width_cm: int
    cell_cm: int
    period_count: int
    period_names: Dict[str, str]
    event_kinds: Tuple[str, ...]
    event_points: Dict[str, int]
    zone_limits: Tuple[float, float, float]
    max_speed_kmh: float
    player_class: int
    ball_class: Optional[int]
    goalkeeper_class: Optional[int]
    referee_class: Optional[int]
    player_model: Path
    pitch_model: Optional[Path]
    tagline: str
    error_kinds: Tuple[str, ...] = ()
    period_targets: Tuple[int, ...] = ()
    win_by: int = 2

    @property
    def zones(self) -> Tuple[str, str, str, str]:
        return ("walk", "jog", "run", "sprint")

    @property
    def nx(self) -> int:
        return int(round(self.court_length_cm / self.cell_cm))

    @property
    def ny(self) -> int:
        return int(round(self.court_width_cm / self.cell_cm))

    @property
    def has_goalkeeper(self) -> bool:
        return self.goalkeeper_class is not None

    @property
    def has_ball(self) -> bool:
        return self.ball_class is not None

    def period_name(self, lang: str = "tr") -> str:
        return self.period_names.get(normalize_lang(lang), self.period_names["tr"])

    def zone_of(self, speed_kmh: float) -> str:
        walk, jog, run = self.zone_limits
        if speed_kmh < walk:
            return "walk"
        if speed_kmh < jog:
            return "jog"
        if speed_kmh < run:
            return "run"
        return "sprint"

    def zone_edges(self) -> Dict[str, float]:
        walk, jog, run = self.zone_limits
        return {"walk": 0.0, "jog": walk, "run": jog, "sprint": run}


def normalize_lang(lang: str) -> str:
    value = str(lang or "").lower()
    if value.startswith("en"):
        return "en"
    if value.startswith("fi"):
        return "fi"
    return "tr"


def _env_path(name: str, default: Optional[Path]) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


FOOTBALL = SportProfile(
    key="football",
    name="Futbol",
    name_en="Football",
    name_fi="Jalkapallo",
    emoji="⚽",
    court_length_cm=12000,
    court_width_cm=7000,
    cell_cm=100,
    period_count=2,
    period_names={"tr": "devre", "fi": "jakso", "en": "half"},
    event_kinds=("goal", "assist"),
    event_points={"goal": 1},
    zone_limits=(7.0, 14.0, 20.0),
    max_speed_kmh=30.0,
    player_class=2,
    ball_class=0,
    goalkeeper_class=1,
    referee_class=3,
    player_model=_env_path("BAKIAPPI_FOOTBALL_MODEL", MODEL_DIR / "football-player-detection.pt"),
    pitch_model=_env_path("BAKIAPPI_FOOTBALL_PITCH_MODEL", MODEL_DIR / "football-pitch-detection.pt"),
    tagline="Halısaha maçını sahaya oturt: mesafe, sprint, ısı haritası.",
)

BASKETBALL = SportProfile(
    key="basketball",
    name="Basketbol",
    name_en="Basketball",
    name_fi="Koripallo",
    emoji="🏀",
    court_length_cm=2800,
    court_width_cm=1500,
    cell_cm=50,
    period_count=4,
    period_names={"tr": "çeyrek", "fi": "neljännes", "en": "quarter"},
    event_kinds=("basket", "basket3", "assist"),
    event_points={"basket": 2, "basket3": 3},
    zone_limits=(6.0, 11.0, 17.0),
    max_speed_kmh=28.0,
    player_class=2,
    ball_class=0,
    goalkeeper_class=None,
    referee_class=None,
    player_model=_env_path(
        "BAKIAPPI_BASKETBALL_MODEL", MODEL_DIR / "football-player-detection.pt"
    ),
    pitch_model=_env_path("BAKIAPPI_BASKETBALL_PITCH_MODEL", None),
    tagline="Kortu köşelerden kalibre et: top, oyuncu, mesafe ve ısı haritası.",
)

VOLLEYBALL = SportProfile(
    key="volleyball",
    name="Voleybol",
    name_en="Volleyball",
    name_fi="Lentopallo",
    emoji="🏐",
    court_length_cm=1800,
    court_width_cm=900,
    cell_cm=25,
    period_count=5,
    period_names={"tr": "set", "fi": "erä", "en": "set"},
    event_kinds=("ace", "kill", "block", "error", "dig", "set", "reception"),
    event_points={"ace": 1, "kill": 1, "block": 1, "error": 1},
    error_kinds=("error",),
    period_targets=(25, 25, 25, 25, 15),
    win_by=2,
    zone_limits=(4.0, 8.0, 13.0),
    max_speed_kmh=26.0,
    # Voleybolda futbol sahası modeli oyuncuları "ball" olarak sınıflıyor.
    # COCO modeli: 0 = person, 32 = sports ball.
    player_class=0,
    ball_class=32,
    goalkeeper_class=None,
    referee_class=None,
    player_model=_env_path(
        "BAKIAPPI_VOLLEYBALL_MODEL", MODEL_DIR / "yolo11n.pt"
    ),
    pitch_model=_env_path("BAKIAPPI_VOLLEYBALL_PITCH_MODEL", None),
    tagline="FIVB kortunu köşelerden kalibre et: servis, blok, top ve set takibi.",
)

PROFILES: Dict[str, SportProfile] = {
    FOOTBALL.key: FOOTBALL,
    BASKETBALL.key: BASKETBALL,
    VOLLEYBALL.key: VOLLEYBALL,
}
SPORTS = tuple(PROFILES.values())


def get(key) -> SportProfile:
    """Branş anahtarını (ya da profilin kendisini) profile çevirir."""
    if isinstance(key, SportProfile):
        return key
    value = str(key or "").strip().lower()
    aliases = {
        "soccer": "football",
        "futbol": "football",
        "basket": "basketball",
        "basketbol": "basketball",
        "volley": "volleyball",
        "voleybol": "volleyball",
    }
    return PROFILES.get(aliases.get(value, value), FOOTBALL)


def is_valid(key: Optional[str]) -> bool:
    return str(key or "").strip().lower() in set(PROFILES) | {
        "soccer",
        "futbol",
        "basket",
        "basketbol",
        "volley",
        "voleybol",
    }


def payload(lang: str = "tr") -> List[dict]:
    """Arayüzdeki branş seçimi için profil listesi."""
    language = normalize_lang(lang)
    rows = []
    for profile in SPORTS:
        name = {"tr": profile.name, "en": profile.name_en, "fi": profile.name_fi}[language]
        rows.append(
            {
                "key": profile.key,
                "name": name,
                "emoji": profile.emoji,
                "tagline": profile.tagline,
                "court": {
                    "length_m": round(profile.court_length_cm / 100, 1),
                    "width_m": round(profile.court_width_cm / 100, 1),
                },
                "periods": profile.period_count,
                "period_name": profile.period_names.get(language, profile.period_names["tr"]),
                "events": list(profile.event_kinds),
                "points": dict(profile.event_points),
                "error_kinds": list(profile.error_kinds),
                "period_targets": list(profile.period_targets),
                "win_by": profile.win_by,
                "scoring": "sets" if profile.key == "volleyball" else "points",
                "zones_kmh": list(profile.zone_limits),
                "calibration": "manual" if profile.pitch_model is None else "auto",
                "has_ball": profile.has_ball,
                "has_goalkeeper": profile.has_goalkeeper,
            }
        )
    return rows


def period_of(profile: SportProfile, elapsed_s: float) -> int:
    """Oyun saatinden içinde bulunulan devre/çeyrek numarasını tahmin eder."""
    minutes = max(0.0, float(elapsed_s or 0.0)) / 60.0
    if profile.key == "basketball":
        length = 15.0
    elif profile.key == "volleyball":
        length = 25.0
    else:
        length = 45.0
    return max(1, min(profile.period_count, int(minutes // length) + 1))


# ------------------------------------------------------------------ kalibrasyon
def court_corners(profile: SportProfile, extent: str = "full") -> List[Tuple[float, float]]:
    """Kalibrasyonda işaretlenen dört köşenin saha koordinatları (cm)."""
    length = profile.court_length_cm
    width = profile.court_width_cm
    if str(extent).lower() == "half":
        length = length / 2.0
    return [(0.0, 0.0), (length, 0.0), (length, width), (0.0, width)]


def calibration_matrix(
    profile: SportProfile, points: Sequence[Sequence[float]], extent: str = "full"
) -> np.ndarray:
    """Görüntüdeki dört köşeden saha koordinatına homografi üretir.

    Sıra: görüntüde sahanın sol-üst, sağ-üst, sağ-alt, sol-alt köşeleri.
    """
    import cv2

    try:
        source = np.array(
            [[float(point[0]), float(point[1])] for point in points], dtype=np.float32
        )
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Kalibrasyon için dört nokta gerekli.") from exc
    if source.shape != (4, 2):
        raise ValueError("Kalibrasyon için dört nokta gerekli.")
    if not np.isfinite(source).all():
        raise ValueError("Kalibrasyon noktaları geçersiz.")
    x = source[:, 0]
    y = source[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if area < 400.0:
        raise ValueError("Seçilen köşeler çok küçük ya da çakışık.")

    target = np.array(court_corners(profile, extent), dtype=np.float32)
    matrix, _ = cv2.findHomography(source, target)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Kalibrasyon homografisi kurulamadı.")
    return matrix


# ---------------------------------------------------------------------- çizim
def _scaled(profile: SportProfile, pixels_per_m: int = PIXELS_PER_M) -> Tuple[int, int, float]:
    width = profile.nx * pixels_per_m
    height = profile.ny * pixels_per_m
    return width, height, width / (profile.court_length_cm / 100.0)



def scoring_team(profile: SportProfile, event: dict) -> int:
    """Olayın sayıyı kazandırdığı takım.

    Olayda ``team`` daima oyuncunun takımıdır. ``error`` olayında sayı
    otomatik olarak rakibe gider; bu kural tek yerde burada uygulanır.
    """
    team = 1 if int(event.get("team") or 0) == 1 else 0
    return 1 - team if str(event.get("kind") or "") in profile.error_kinds else team


def volleyball_sets(profile: SportProfile, events: Sequence[dict]) -> dict:
    """Olay sırasına göre FIVB set skorunu hesaplar.

    Setler 25 sayı (5. set 15) ve en az 2 fark kuralıyla kapanır; 3 set
    kazanan maçı alır. Hata olayının rakibe sayı yazması ``scoring_team``
    içinde, tek bir yerde uygulanır.
    """
    targets = list(profile.period_targets) or [25] * max(0, profile.period_count - 1) + [15]
    sets: List[dict] = []
    wins = [0, 0]
    current = [0, 0]
    complete = False
    for event in sorted(
        events,
        key=lambda row: (
            float(row.get("minute") or 0.0),
            int(row.get("id") or 0),
        ),
    ):
        kind = str(event.get("kind") or "")
        points = int(profile.event_points.get(kind, 0))
        if points <= 0:
            continue
        team = scoring_team(profile, event)
        current[team] += points
        target = targets[min(len(sets), len(targets) - 1)]
        other = 1 - team
        if current[team] >= target and current[team] - current[other] >= profile.win_by:
            sets.append(
                {
                    "set": len(sets) + 1,
                    "a": current[0],
                    "b": current[1],
                    "winner": "A" if team == 0 else "B",
                }
            )
            wins[team] += 1
            current = [0, 0]
            if wins[team] >= 3 or len(sets) >= profile.period_count:
                complete = True
                break
    return {
        "sets": sets,
        "current": current,
        "wins": wins,
        "number": min(len(sets) + 1, profile.period_count),
        "complete": complete,
        "targets": targets,
    }

def court_image(profile: SportProfile, pixels_per_m: int = PIXELS_PER_M) -> Image.Image:
    """Branşa göre saha/kort görseli (ısı haritası ve radar için ortak)."""
    width, height, m = _scaled(profile, pixels_per_m)
    image = Image.new("RGB", (width, height), COURT_FILL)
    draw = ImageDraw.Draw(image)
    if profile.key == "basketball":
        _draw_basketball(draw, m, profile.court_length_cm / 100.0, profile.court_width_cm / 100.0)
    elif profile.key == "volleyball":
        image = Image.new("RGB", (width, height), (10, 24, 38))
        draw = ImageDraw.Draw(image)
        _draw_volleyball(draw, m, profile.court_length_cm / 100.0, profile.court_width_cm / 100.0)
    else:
        _draw_football(draw, m)
    return image


def _draw_football(draw: ImageDraw.ImageDraw, m: float) -> None:
    def box(x0, y0, x1, y1):
        return (x0 * m, y0 * m, x1 * m, y1 * m)

    draw.rectangle(box(0.4, 0.4, 119.6, 69.6), outline=COURT_LINE, width=4)
    draw.line([(60 * m, 0.4 * m), (60 * m, 69.6 * m)], fill=COURT_LINE, width=4)
    radius = 9.15 * m
    draw.ellipse(
        (60 * m - radius, 35 * m - radius, 60 * m + radius, 35 * m + radius),
        outline=COURT_LINE,
        width=4,
    )
    draw.ellipse(box(59.5, 34.5, 60.5, 35.5), fill=COURT_LINE)
    for left in (True, False):
        if left:
            penalty = (0.4, 20.15)
            goal_area = (0.4, 5.5)
            spot = 11.0
        else:
            penalty = (99.85, 119.6)
            goal_area = (114.5, 119.6)
            spot = 109.0
        draw.rectangle(box(penalty[0], 14.5, penalty[1], 55.5), outline=COURT_LINE, width=4)
        draw.rectangle(box(goal_area[0], 25.84, goal_area[1], 44.16), outline=COURT_LINE, width=4)
        draw.ellipse(box(spot - 0.4, 34.6, spot + 0.4, 35.4), fill=COURT_LINE)


def _draw_basketball(draw: ImageDraw.ImageDraw, m: float, length: float, width: float) -> None:
    """FIBA ölçüleriyle 28 × 15 m kort (m: piksel/metre)."""
    line = COURT_LINE
    thickness = 3

    def xy(x: float, y: float) -> Tuple[float, float]:
        return (x * m, y * m)

    def box(x0, y0, x1, y1):
        return (x0 * m, y0 * m, x1 * m, y1 * m)

    draw.rectangle(box(0.6, 0.6, length - 0.6, width - 0.6), outline=line, width=thickness)
    draw.line([xy(length / 2, 0.6), xy(length / 2, width - 0.6)], fill=line, width=thickness)

    centre = (length / 2, width / 2)
    for radius in (1.8,):
        draw.ellipse(
            (centre[0] * m - radius * m, centre[1] * m - radius * m,
             centre[0] * m + radius * m, centre[1] * m + radius * m),
            outline=line,
            width=thickness,
        )

    basket_from_baseline = 1.575
    lane_half = 2.45
    lane_length = 5.8
    for side in (0, 1):
        sign = 1 if side == 0 else -1
        base_x = 0.6 if side == 0 else length - 0.6
        basket_x = basket_from_baseline if side == 0 else length - basket_from_baseline
        free_throw_x = lane_length if side == 0 else length - lane_length

        # serbest atış koridoru
        draw.rectangle(
            box(
                min(base_x, free_throw_x),
                width / 2 - lane_half,
                max(base_x, free_throw_x),
                width / 2 + lane_half,
            ),
            outline=line,
            width=thickness,
        )
        # serbest atış yarım dairesi (koridor dışına bakan yay)
        draw.arc(
            (
                free_throw_x * m - 1.8 * m,
                (width / 2 - 1.8) * m,
                free_throw_x * m + 1.8 * m,
                (width / 2 + 1.8) * m,
            ),
            start=270 if side == 0 else 90,
            end=90 if side == 0 else 270,
            fill=line,
            width=thickness,
        )
        # üç sayı çizgisi: köşe düzleri + yay
        straight_end = basket_x + sign * 1.575
        draw.line(
            [xy(basket_x, 0.9), xy(straight_end, 0.9)], fill=line, width=thickness
        )
        draw.line(
            [xy(basket_x, width - 0.9), xy(straight_end, width - 0.9)],
            fill=line,
            width=thickness,
        )
        angle = math.degrees(math.asin(max(-1.0, min(1.0, (width / 2 - 0.9) / 6.75))))
        if side == 0:
            draw.arc(
                (
                    basket_x * m - 6.75 * m,
                    (width / 2 - 6.75) * m,
                    basket_x * m + 6.75 * m,
                    (width / 2 + 6.75) * m,
                ),
                start=-angle,
                end=angle,
                fill=line,
                width=thickness,
            )
        else:
            draw.arc(
                (
                    basket_x * m - 6.75 * m,
                    (width / 2 - 6.75) * m,
                    basket_x * m + 6.75 * m,
                    (width / 2 + 6.75) * m,
                ),
                start=180 - angle,
                end=180 + angle,
                fill=line,
                width=thickness,
            )
        # hücum faul yayı ve pota
        draw.arc(
            (
                basket_x * m - 1.25 * m,
                (width / 2 - 1.25) * m,
                basket_x * m + 1.25 * m,
                (width / 2 + 1.25) * m,
            ),
            start=270 if side == 0 else 90,
            end=90 if side == 0 else 270,
            fill=line,
            width=2,
        )
        draw.ellipse(
            (
                basket_x * m - 0.23 * m,
                width / 2 * m - 0.23 * m,
                basket_x * m + 0.23 * m,
                width / 2 * m + 0.23 * m,
            ),
            outline=(255, 170, 90),
            width=2,
        )
        draw.line(
            [xy(basket_x, width / 2 - 0.9), xy(basket_x, width / 2 + 0.9)],
            fill=(200, 150, 100),
            width=3,
        )


def radar_bgr(profile: SportProfile, width: int, height: int, padding: int = 8) -> np.ndarray:
    """Video radarı için kort görselini istenen boyuta getirip BGR döndürür."""
    import cv2

    image = court_image(profile)
    target_w = max(8, width - padding * 2)
    target_h = max(8, height - padding * 2)
    ratio = min(target_w / image.width, target_h / image.height)
    size = (max(8, int(image.width * ratio)), max(8, int(image.height * ratio)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), COURT_FILL)
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Homografiyi noktalara uygular (santimetre)."""
    import cv2

    shaped = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2)

def _draw_volleyball(
    draw: ImageDraw.ImageDraw, m: float, length: float, width: float
) -> None:
    """FIVB 18 × 9 m voleybol kortu: hücum çizgileri, file ve servis işaretleri."""
    line = (125, 196, 232)
    net = (255, 172, 86)
    thickness = 3

    def xy(x: float, y: float) -> Tuple[float, float]:
        return (x * m, y * m)

    def box(x0, y0, x1, y1):
        return (x0 * m, y0 * m, x1 * m, y1 * m)

    # Oyun alanı ve serbest bölge sınırı.
    draw.rectangle(box(0.5, 0.5, length - 0.5, width - 0.5), outline=line, width=thickness)
    # Orta çizgi + file (üstten görünümde orta çizgiye binmiş ağ).
    draw.line([xy(length / 2, 0.5), xy(length / 2, width - 0.5)], fill=line, width=thickness)
    draw.line([xy(length / 2, 0.15), xy(length / 2, width - 0.15)], fill=net, width=5)
    # FIVB hücum çizgileri: orta çizgiden 3 m uzakta.
    for x in (length / 2 - 3.0, length / 2 + 3.0):
        draw.line([xy(x, 0.5), xy(x, width - 0.5)], fill=line, width=thickness)
    # File direkleri / antenler.
    for y in (0.15, width - 0.15):
        draw.ellipse(
            (length / 2 * m - 0.12 * m, y * m - 0.12 * m,
             length / 2 * m + 0.12 * m, y * m + 0.12 * m),
            fill=net,
        )
    # Servis bölgesi çizgileri (arka çizginin 20 cm gerisinde kısa işaretler).
    for side in (0, 1):
        base = 0.3 if side == 0 else length - 0.3
        for y in (width * 0.25, width * 0.75):
            draw.line([xy(base, y - 0.18), xy(base, y + 0.18)], fill=line, width=thickness)

