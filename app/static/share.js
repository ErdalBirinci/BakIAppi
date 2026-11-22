/* BakIAppi — herkese açık maç sayfası (salt okunur). */

const SHARE_TOKEN = location.pathname.split("/").filter(Boolean).pop() || "";
const $ = (id) => document.getElementById(id);
let shareData = null;
let shareTrendData = null;

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[ch]);
}

function distance(value) {
  const metres = Number(value) || 0;
  return metres >= 1000 ? `${(metres / 1000).toFixed(2)} km` : `${Math.round(metres)} m`;
}

function speed(value) {
  return `${(Number(value) || 0).toFixed(1)} km/s`;
}

function render() {
  if (!shareData) return;
  const teams = shareData.teams || {};
  const players = (shareData.players || [])
    .slice()
    .sort((a, b) => (Number(b.distance_m) || 0) - (Number(a.distance_m) || 0));

  const status = shareData.status || "finished";
  $("shareGroup").textContent = shareData.group || "";
  $("shareTitle").textContent = shareData.title || "";

  const pill = $("shareStatus");
  pill.hidden = false;
  pill.className = `pill ${status}`;
  pill.textContent =
    status === "live"
      ? shareData.paused
        ? t("share.status.paused")
        : t("share.status.live")
      : t("share.status.finished");

  const sport = shareData.sport || "football";
  const basketball = sport === "basketball";
  const volleyball = sport === "volleyball";
  const minutes = Math.round((shareData.elapsed_s || 0) / 60);
  const period =
    status === "live" && shareData.period
      ? t(
          basketball ? "period.quarter" : volleyball ? "period.set" : "period.half",
          { n: shareData.period }
        )
      : "";
  $("shareMeta").textContent = [shareData.date, `${minutes} ${t("share.minute")}`, period]
    .filter(Boolean)
    .join(" · ");

  const topSpeed = Math.max(
    Number(teams["0"]?.top_speed_kmh) || 0,
    Number(teams["1"]?.top_speed_kmh) || 0
  );
  const events = shareData.events || [];
  const sets = volleyball ? shareData.sets : null;
  const scoringKinds = basketball
    ? { basket: 2, basket3: 3 }
    : volleyball
      ? { ace: 1, kill: 1, block: 1, error: 1 }
      : { goal: 1 };
  const goals = events.filter((event) => scoringKinds[event.kind]);
  const score = sets
    ? sets.wins
    : goals.reduce(
        (total, event) => {
          total[Number(event.team) === 1 ? 1 : 0] += scoringKinds[event.kind] || 0;
          return total;
        },
        [0, 0]
      );
  $("shareCards").innerHTML = [
    `<div class="card"><b>${esc(
      t(basketball ? "sport.basketball" : volleyball ? "sport.volleyball" : "sport.football")
    )}</b><span>${esc(t("sport.current"))}</span></div>`,
    `<div class="card"><b>${players.length}</b><span>${esc(t("stat.squad"))}</span></div>`,
    `<div class="card t0"><b>${distance(teams["0"]?.distance_m)}</b><span>${esc(t("stat.teamADistance"))}</span></div>`,
    `<div class="card t1"><b>${distance(teams["1"]?.distance_m)}</b><span>${esc(t("stat.teamBDistance"))}</span></div>`,
    `<div class="card"><b>${speed(topSpeed)}</b><span>${esc(t("stat.topSpeed"))}</span></div>`,
    `<div class="card"><b>${score[0]} – ${score[1]}</b><span>${esc(
      t(
        basketball
          ? "events.titleBasketball"
          : volleyball
            ? "events.titleVolleyball"
            : "events.title"
      )
    )}</span></div>`,
  ].join("");

  const scorers = $("shareScorers");
  let setText = "";
  if (sets) {
    setText = t("events.setsScore", { a: sets.wins[0], b: sets.wins[1] });
    if (!sets.complete) {
      setText += ` · ${t("events.setScore", {
        set: sets.number,
        a: sets.current[0],
        b: sets.current[1],
      })}`;
    }
  }
  if (goals.length || setText) {
    scorers.textContent = [
      setText,
      ...goals.map(
        (event) => `${Math.round(Number(event.minute) || 0)}' ${event.name || `#${event.tracker_id}`}`
      ),
    ]
      .filter(Boolean)
      .join(" · ");
    scorers.hidden = false;
  } else {
    scorers.hidden = true;
  }

  $("shareRows").innerHTML = players
    .map((row) => {
      const team = Number(row.team) === 1 ? 1 : 0;
      const name = row.name || `#${row.tracker_id}`;
      return `<tr>
        <td>${esc(name)}</td>
        <td class="team${team}">${esc(team === 0 ? t("heat.teamA") : t("heat.teamB"))}</td>
        <td>${distance(row.distance_m)}</td>
        <td>${speed(row.top_speed_kmh)}</td>
        <td>${speed(row.avg_speed_kmh)}</td>
        <td>${row.sprints ?? 0}</td>
      </tr>`;
    })
    .join("");
  $("shareTable").hidden = !players.length;
}

function renderTrend() {
  if (!shareTrendData) return;
  const drawn = TrendChart.render($("shareTrend"), $("shareTrendLegend"), {
    matches: shareTrendData.matches || [],
    series: shareTrendData.series || [],
    valueOf: (point) => point.distance_m,
    unit: t("trend.unit.distance_m"),
  });
  $("shareTrendPanel").hidden = !drawn;
}

let pollTimer = null;

function schedulePoll() {
  clearTimeout(pollTimer);
  if (shareData && shareData.status === "live") {
    pollTimer = setTimeout(refresh, 8000);
  }
}

async function refresh() {
  try {
    const response = await fetch(`/api/public/${encodeURIComponent(SHARE_TOKEN)}`);
    if (!response.ok) throw new Error(t("share.invalid"));
    shareData = await response.json();
    render();
    schedulePoll();
  } catch {
    pollTimer = setTimeout(refresh, 15000);
  }
}

async function load() {
  try {
    const response = await fetch(`/api/public/${encodeURIComponent(SHARE_TOKEN)}`);
    if (!response.ok) throw new Error(t("share.invalid"));
    shareData = await response.json();
    document.title = `${shareData.title} — BakIAppi`;
    render();
    schedulePoll();

    const heat = $("shareHeat");
    heat.onload = () => {
      $("shareHeatPanel").hidden = false;
    };
    heat.onerror = () => {
      heat.removeAttribute("src");
      $("shareHeatPanel").hidden = true;
    };
    heat.src = `/api/public/${encodeURIComponent(SHARE_TOKEN)}/heatmap.png?scope=all`;

    try {
      const trendResponse = await fetch(
        `/api/public/${encodeURIComponent(SHARE_TOKEN)}/trends`
      );
      if (trendResponse.ok) {
        shareTrendData = await trendResponse.json();
        renderTrend();
      }
    } catch {
      // Grafik opsiyonel; sayfanın geri kalanı çalışsın.
    }
  } catch (error) {
    $("shareErr").hidden = false;
    $("shareErr").textContent = serverMessage(error.message);
  }
}

document.addEventListener("i18n:changed", () => {
  render();
  renderTrend();
});
load();
