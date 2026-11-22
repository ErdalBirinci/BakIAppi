"""BakIAppi — saha/kort ısı haritası: yoğunluk toplama ve PNG üretimi.

Konumlar `MotionTracker.corrected` içinden santimetre cinsinden saha koordinatı
olarak gelir. Izgara ve zemin çizimi branşa göre `app.sports` profilinden alınır:
futbol 120 × 70 m / 1 m hücre, basketbol 28 × 15 m / 0,5 m hücre,
voleybol 18 × 9 m / 0,25 m hücre.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app import sports

PIXELS_PER_M = 10

CACHE_DIR = Path(__file__).resolve().parent / "data" / "heatmaps"

_STOPS = (
    (0.00, (0, 120, 200)),
    (0.40, (0, 191, 255)),
    (0.70, (255, 20, 147)),
    (1.00, (255, 215, 0)),
)

# Geriye dönük: futbol ızgarası.
_PROFILE = sports.FOOTBALL
NX, NY = _PROFILE.nx, _PROFILE.ny
CELL_M = _PROFILE.cell_cm / 100.0
PITCH_CM_X, PITCH_CM_Y = _PROFILE.court_length_cm, _PROFILE.court_width_cm
WIDTH, HEIGHT = _PROFILE.nx * PIXELS_PER_M, _PROFILE.ny * PIXELS_PER_M


def _profile(sport) -> sports.SportProfile:
    return sport if isinstance(sport, sports.SportProfile) else sports.get(sport)


def cell_of(position_cm: Iterable[float], sport=None) -> Optional[int]:
    """Saha koordinatını (cm) hücre numarasına çevirir; saha dışı None."""
    profile = _profile(sport)
    x, y = float(position_cm[0]), float(position_cm[1])
    if not (0.0 <= x < profile.court_length_cm and 0.0 <= y < profile.court_width_cm):
        return None
    return int(y // profile.cell_cm) * profile.nx + int(x // profile.cell_cm)


def accumulate(bucket: dict[int, int], position_cm: Iterable[float], sport=None) -> None:
    """Bir konumu sayaca ekler (saha dışı konumlar yok sayılır)."""
    cell = cell_of(position_cm, sport)
    if cell is not None:
        bucket[cell] = bucket.get(cell, 0) + 1


def merge(buckets: Iterable[dict[int, int]]) -> dict[int, int]:
    total: dict[int, int] = {}
    for bucket in buckets:
        for cell, count in bucket.items():
            total[cell] = total.get(cell, 0) + int(count)
    return total


def _pitch(profile: sports.SportProfile) -> Image.Image:
    return sports.court_image(profile)


def _ramp(values: np.ndarray) -> np.ndarray:
    positions = np.array([stop[0] for stop in _STOPS], dtype=np.float32)
    colors = np.array([stop[1] for stop in _STOPS], dtype=np.float32)
    out = np.zeros(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(values, positions, colors[:, channel])
    return out


def _density(bucket: dict[int, int], profile: sports.SportProfile) -> Optional[Image.Image]:
    nx, ny = profile.nx, profile.ny
    grid = np.zeros((ny, nx), dtype=np.float32)
    for key, count in bucket.items():
        try:
            cell = int(key)
        except (TypeError, ValueError):
            continue
        y, x = divmod(cell, nx)
        if 0 <= y < ny and 0 <= x < nx:
            grid[y, x] += float(count)

    if float(grid.max()) <= 0.0:
        return None

    width = nx * PIXELS_PER_M
    height = ny * PIXELS_PER_M
    layer = Image.fromarray((grid / grid.max() * 255.0).astype(np.uint8), "L")
    layer = layer.resize((width, height), Image.Resampling.BICUBIC)
    layer = layer.filter(ImageFilter.GaussianBlur(16 if profile.key == "football" else 10))

    values = np.asarray(layer, dtype=np.float32) / 255.0
    peak = float(np.percentile(values, 99.0))
    if peak <= 0.0:
        peak = float(values.max()) or 1.0
    values = np.clip(values / peak, 0.0, 1.0) ** 0.85

    rgb = _ramp(values)
    alpha = np.clip((values - 0.05) / 0.95, 0.0, 1.0) ** 0.7
    rgba = np.dstack([rgb, alpha * 240.0]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def focus_box(
    cells: dict[int, int],
    aspect: float = 220 / 128,
    pad: float = 0.45,
    min_width: float = 0.32,
    sport=None,
) -> Optional[tuple[float, float, float, float]]:
    """Yoğunluğun olduğu bölgeyi, istenen en-boy oranında pencereye çevirir.

    Dönen değer saha içinde normalize (sol, üst, sağ, alt) aralığıdır; veri yoksa None.
    """
    profile = _profile(sport)
    nx, ny = profile.nx, profile.ny
    if not cells:
        return None
    xs = [int(cell) % nx for cell in cells]
    ys = [int(cell) // nx for cell in cells]
    x0, x1 = min(xs) / nx, (max(xs) + 1) / nx
    y0, y1 = min(ys) / ny, (max(ys) + 1) / ny

    pad_x, pad_y = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1 = max(0.0, x0 - pad_x), min(1.0, x1 + pad_x)
    y0, y1 = max(0.0, y0 - pad_y), min(1.0, y1 + pad_y)

    ratio = aspect * (ny / nx)  # normalize genişlik / yükseklik
    width = max(x1 - x0, min_width)
    height = width / ratio
    if height > 1.0:
        height = 1.0
        width = min(height * ratio, 1.0)

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - width / 2, 0.0), 1.0 - width)
    top = min(max(cy - height / 2, 0.0), 1.0 - height)
    return (round(left, 4), round(top, 4), round(left + width, 4), round(top + height, 4))


def render(bucket: dict[int, int], sport=None) -> Optional[bytes]:
    """Yoğunluk haritasını saha üzerine bindirip PNG döndürür; veri yoksa None."""
    profile = _profile(sport)
    layer = _density(bucket, profile)
    if layer is None:
        return None

    image = Image.alpha_composite(_pitch(profile).convert("RGBA"), layer).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
