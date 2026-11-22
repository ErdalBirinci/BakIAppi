/* Ortak trend grafiği: eksenler, noktalar ve dokunmatik/fare ipucu. */

const TrendChart = (() => {
  const PALETTE = ["#ff3d81", "#35d0ba", "#ffd166", "#7c9cff", "#c084fc"];
  const WIDTH = 640;
  const HEIGHT = 200;
  const PAD = { left: 58, right: 18, top: 14, bottom: 30 };

  function escapeAttr(value) {
    return String(value ?? "").replace(
      /[&<>"']/g,
      (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]
    );
  }

  function shortDate(timestamp) {
    if (!timestamp) return "—";
    const date = new Date(timestamp * 1000);
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const lang = typeof currentLang === "function" ? currentLang() : "tr";
    if (lang === "fi") return `${day}.${month}.`;
    if (lang === "en") return `${day}/${month}`;
    return `${day}.${month}`;
  }

  function valueText(value, unit) {
    const number = Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
    return `${number}${unit ? ` ${unit}` : ""}`;
  }

  // Grafik başına ipucu kapatıcısı; hem bağlama hem render temizliği buradan.
  const tipHiders = new WeakMap();

  function bindTooltip(svg) {
    if (tipHiders.has(svg)) return;

    const MARGIN = 8;
    const GAP = 12;
    const tip = document.createElement("div");
    tip.className = "trendtip";
    tip.hidden = true;
    document.body.appendChild(tip);

    let active = null;
    let hideTimer = null;
    const hide = () => {
      clearTimeout(hideTimer);
      active = null;
      tip.hidden = true;
    };
    tipHiders.set(svg, hide);
    const place = () => {
      if (!active) return;
      const rect = active.getBoundingClientRect();
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      let left = rect.left + rect.width / 2 - width / 2;
      let top = rect.top - height - GAP;
      if (top < MARGIN) top = rect.bottom + GAP;
      left = Math.min(Math.max(left, MARGIN), window.innerWidth - MARGIN - width);
      top = Math.min(Math.max(top, MARGIN), window.innerHeight - MARGIN - height);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };
    const show = (event) => {
      const point = event.target.closest("[data-tip]");
      if (!point) return hide();
      clearTimeout(hideTimer);
      active = point;
      tip.textContent = point.dataset.tip;
      tip.hidden = false;
      place();
    };

    svg.addEventListener("pointermove", show);
    svg.addEventListener("pointerdown", show);
    svg.addEventListener("pointercancel", hide);
    // Dokunmatikte parmak kalkınca pointerleave gelir; ipucu bir süre kalsın.
    svg.addEventListener("pointerleave", (event) => {
      if (event.pointerType !== "touch") return hide();
      clearTimeout(hideTimer);
      hideTimer = setTimeout(hide, 3500);
    });
    document.addEventListener("pointerdown", (event) => {
      if (!svg.contains(event.target)) hide();
    });
    // Kaydırma/yeniden boyutlamada ipucu, bağlı olduğu noktayla birlikte kalır.
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, { passive: true });
  }

  /* Çizer; grafik gösterildiyse true döner. Eksen birimi dile göre gelir. */
  function render(svg, legend, options) {
    tipHiders.get(svg)?.();
    const { matches = [], series = [], valueOf, unit = "", wholeNumbers = false } = options;
    const entries = series
      .map((row) => ({
        row,
        values: row.points.filter((point) => Number.isFinite(valueOf(point))),
      }))
      .filter((entry) => entry.values.length >= 2)
      .slice(0, 5);

    if (matches.length < 2 || !entries.length) {
      svg.innerHTML = "";
      if (legend) legend.innerHTML = "";
      return false;
    }

    const innerWidth = WIDTH - PAD.left - PAD.right;
    const innerHeight = HEIGHT - PAD.top - PAD.bottom;
    const peak = Math.max(
      ...entries.flatMap((entry) => entry.values.map((point) => valueOf(point))),
      0
    );
    const scale = peak > 0 ? peak : 1;
    const px = (index) => PAD.left + index * (innerWidth / (matches.length - 1));
    const py = (value) => HEIGHT - PAD.bottom - (value / scale) * innerHeight;

    const ratios = peak <= 0 ? [0] : wholeNumbers ? [1, 0] : [1, 0.5, 0];
    const grid = ratios
      .map((ratio) => {
        const tick = peak * ratio;
        const lineY = py(tick).toFixed(1);
        const label = valueText(tick, ratio === 1 || peak <= 0 ? unit : "");
        return `<line x1="${PAD.left}" y1="${lineY}" x2="${
          WIDTH - PAD.right
        }" y2="${lineY}" stroke="rgba(255,255,255,.14)" stroke-width="1" />
        <text class="axis" x="${PAD.left - 8}" y="${lineY}" text-anchor="end" dominant-baseline="middle">${label}</text>`;
      })
      .join("");

    const labelStep = Math.ceil(matches.length / 6);
    const xLabels = matches
      .map((match, index) => ({ match, index }))
      .filter(({ index }) => index % labelStep === 0 || index === matches.length - 1)
      .map(
        ({ match, index }) =>
          `<text class="axis" x="${px(index).toFixed(
            1
          )}" y="${HEIGHT - 9}" text-anchor="middle">${shortDate(
            match.started_at
          )}</text>`
      )
      .join("");

    const lines = entries.map(({ row }, index) => {
      const color = PALETTE[index % PALETTE.length];
      const dots = [];
      let path = "";
      let pen = false;
      row.points.forEach((point, i) => {
        const value = valueOf(point);
        if (!Number.isFinite(value)) {
          pen = false;
          return;
        }
        const cx = px(i).toFixed(1);
        const cy = py(value).toFixed(1);
        path += `${pen ? "L" : "M"}${cx} ${cy} `;
        pen = true;
        const tooltip = `${shortDate(matches[i].started_at)} · ${valueText(value, unit)}`;
        dots.push(
          `<g class="trendpoint" data-tip="${escapeAttr(
            tooltip
          )}"><circle class="hit" cx="${cx}" cy="${cy}" r="9" /><circle cx="${cx}" cy="${cy}" r="3" fill="${color}" /></g>`
        );
      });
      return `<path d="${path.trim()}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />${dots.join(
        ""
      )}`;
    });

    svg.innerHTML = grid + xLabels + lines.join("");
    if (legend) {
      legend.innerHTML = entries
        .map(
          ({ row }, index) =>
            `<span class="trendchip"><i style="background:${
              PALETTE[index % PALETTE.length]
            }"></i>${escapeAttr(row.name)}</span>`
        )
        .join("");
    }
    bindTooltip(svg);
    return true;
  }

  return { render };
})();
