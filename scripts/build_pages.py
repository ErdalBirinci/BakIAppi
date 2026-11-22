#!/usr/bin/env python3
"""Build the static BakIAppi landing page for GitHub Pages."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "static"

STATIC_LINKS = (
    ('href="/login.html">Panele gir', 'href="#branslar">Dört branşı keşfet'),
    ('href="/login.html">\n          Analizi başlat', 'href="#nasil">\n          Nasıl çalışır'),
    ('href="/login.html">Kendi maçını dene', 'href="#branslar">Branşları keşfet'),
    ('href="/login.html">Giriş', 'href="#branslar">Branşlar'),
    ('href="/studio.html">Video ile dene', 'href="#nasil">Nasıl çalışır'),
    ('href="/studio.html">Video analizi', 'href="#nasil">Nasıl çalışır'),
)


def build(destination: Path) -> None:
    destination = destination.resolve()
    if destination == SOURCE or SOURCE in destination.parents:
        raise SystemExit("Output directory must not be inside app/static.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in ("landing.css", "landing.js"):
        shutil.copy2(SOURCE / name, destination / name)

    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    app_url = os.environ.get("BAKIAPPI_APP_URL", "").strip()
    if app_url and not app_url.startswith(("http://", "https://")):
        raise SystemExit("BAKIAPPI_APP_URL must start with http:// or https://.")

    # Project Pages live under /BakIAppi/, so public assets must stay relative.
    html = html.replace('href="/landing.css"', 'href="./landing.css"')
    html = html.replace('src="/landing.js"', 'src="./landing.js"')
    html = html.replace('<a class="brand" href="/"', '<a class="brand" href="./"')

    if app_url:
        # The panel is a separate app; send all app actions there when configured.
        for route in ("/login.html", "/studio.html"):
            html = html.replace(f'href="{route}"', f'href="{app_url}"')
    else:
        # Without an app URL, make every action stay inside the static landing.
        html = html.replace(
            '    <a class="btn btn-ghost" href="/studio.html">Video analizi</a>\n',
            "",
        )
        for old, new in STATIC_LINKS:
            html = html.replace(old, new)

    if 'href="/landing.css"' in html or 'src="/landing.js"' in html:
        raise SystemExit("Landing still references root-relative assets.")
    if 'href="/login.html"' in html or 'href="/studio.html"' in html:
        raise SystemExit("Landing still references app-only routes.")
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
    build(destination)
    print(f"Built landing in {destination}")


if __name__ == "__main__":
    main()
