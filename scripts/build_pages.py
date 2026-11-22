#!/usr/bin/env python3
"""Build the static BakIAppi landing page for GitHub Pages."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "static"
DEFAULT_APP_URL = "https://github.com/ErdalBirinci/BakIAppi"


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
    app_url = os.environ.get("BAKIAPPI_APP_URL", "").strip() or DEFAULT_APP_URL

    # Project Pages live under /BakIAppi/, so public assets must stay relative.
    html = html.replace('href="/landing.css"', 'href="./landing.css"')
    html = html.replace('src="/landing.js"', 'src="./landing.js"')
    html = html.replace('<a class="brand" href="/"', '<a class="brand" href="./"')

    # The analysis panel is a local Python app, not a GitHub Pages backend.
    for route in ("/login.html", "/studio.html"):
        html = html.replace(f'href="{route}"', f'href="{app_url}"')

    if 'href="/landing.css"' in html or 'src="/landing.js"' in html:
        raise SystemExit("Landing still references root-relative assets.")
    (destination / "index.html").write_text(html, encoding="utf-8")
    (destination / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"
    build(destination)
    print(f"Built landing in {destination}")


if __name__ == "__main__":
    main()
