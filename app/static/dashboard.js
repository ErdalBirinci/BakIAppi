const $ = (id) => document.getElementById(id);

let liveId = null;
let pollTimer = null;
let lastLive = null;
let lastDetail = null;
let lastPlayers = [];
const TREND_METRICS = ["distance_m", "top_speed_kmh", "sprints"];

const TREND_METRIC_KEY = "bakiappi.trendMetric";
let trendMetric = loadTrendMetric();
let trendData = null;

function loadTrendMetric() {
  try {
    const stored = localStorage.getItem(TREND_METRIC_KEY);
    return TREND_METRICS.includes(stored) ? stored : TREND_METRICS[0];
  } catch {
    return TREND_METRICS[0];
  }
}

function saveTrendMetric(value) {
  try {
    localStorage.setItem(TREND_METRIC_KEY, value);
  } catch {
    /* özel pencere: tercih yalnızca bu oturumda kalır */
  }
}
let cardVersion = 0;
let liveHeatId = null;
let liveTick = 0;
let liveEventsMatch = null;
let livePlayersKey = "";
const draftNames = new Map();
const heatState = {};

let profiles = [];
let groupSport = "football";
let currentProfile = null;
const calib = { match: null, corners: [], image: null, extent: "full" };

const STATUS_KEY = {
  connecting: "status.connecting",
  live: "status.live",
  finished: "status.finished",
  error: "status.error",
};

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]
  );

function showError(message) {
  const box = $("err");
  box.textContent = serverMessage(String(message));
  box.hidden = false;
}

let toastTimer = null;
function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    box.hidden = true;
  }, 4200);
}

async function api(url, options) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    location.href = "/login.html";
    throw new Error(t("msg.loginRequired"));
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || response.statusText);
  return payload;
}

function clock(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

function dateText(timestamp) {
  if (!timestamp) return "—";
  const locale = { tr: "tr-TR", fi: "fi-FI", en: "en-GB" }[currentLang()] || "tr-TR";
  return new Date(timestamp * 1000).toLocaleString(locale, {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function card(value, label, className = "") {
  return `<div class="card ${className}"><b>${value}</b><span>${esc(label)}</span></div>`;
}

function teamOf(stats, index) {
  return (stats?.teams || {})[String(index)] || { players: 0, distance_m: 0, top_speed_kmh: 0 };
}

const FALLBACK_PROFILE = {
  key: "football",
  periods: 2,
  events: ["goal", "assist"],
  points: { goal: 1 },
  has_ball: false,
  calibration: "auto",
};

function profileOf(key) {
  return profiles.find((row) => row.key === key) || profiles[0] || FALLBACK_PROFILE;
}

function sportName(key) {
  const profile = profileOf(key);
  const nameKey = { basketball: "sport.basketball", volleyball: "sport.volleyball" }[profile.key];
  return t(nameKey || "sport.football");
}

function periodLabel(number, sport) {
  const key = { basketball: "period.quarter", volleyball: "period.set" }[sport] || "period.half";
  return t(key, { n: number });
}

function playerRows(players) {
  if (!players || !players.length) {
    return `<tr><td colspan="7" class="muted">${esc(t("msg.noPlayers"))}</td></tr>`;
  }
  return players
    .slice(0, 40)
    .map(
      (row) => `<tr>
        <td>${row.name ? `${esc(row.name)} <span class="muted">#${row.tracker_id}</span>` : `#${row.tracker_id}`}</td>
        <td class="team${row.team}">${row.team === 0 ? "A" : "B"}</td>
        <td>${row.distance_m} m</td>
        <td>${row.top_speed_kmh} km/s</td>
        <td>${row.avg_speed_kmh || 0} km/s</td>
        <td>${row.sprints ?? 0}</td>
        <td>${row.samples || "—"}</td>
      </tr>`
    )
    .join("");
}

/* --------------------------------------------------------------- isim atama */
async function loadNames(match) {
  try {
    const payload = await api(`/api/matches/${match.id}/names`);
    return payload.names || {};
  } catch {
    return {};
  }
}

function renderNames(players) {
  const grid = $("nameGrid");
  if (!players.length) {
    grid.innerHTML = "";
    return;
  }
  grid.innerHTML = players
    .map((row) => {
      const value = draftNames.has(row.tracker_id) ? draftNames.get(row.tracker_id) : row.name || "";
      return `<label class="namerow">
        <span>#${row.tracker_id} · ${row.distance_m} m</span>
        <input data-tid="${row.tracker_id}" value="${esc(value)}" maxlength="40" list="rosterNames"
               placeholder="${esc(t("names.placeholder"))}">
      </label>`;
    })
    .join("");
}

async function saveNames() {
  if (!lastDetail) return;
  const names = {};
  $("nameGrid")
    .querySelectorAll("input[data-tid]")
    .forEach((input) => {
      const value = input.value.trim();
      if (value) names[input.dataset.tid] = value;
    });
  try {
    await api(`/api/matches/${lastDetail.id}/names`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    draftNames.clear();
    cardVersion += 1; // tarayıcı eski kart görselini yeniden kullanmasın
    await renderDetail(lastDetail, false);
    await loadRoster();
    toast(t("names.saved"));
  } catch (error) {
    showError(error.message);
  }
}

/* --------------------------------------------------------------- ısı haritası */
function heatOptions(players, isLive) {
  const options = [
    ["all", t("heat.all")],
    ["team0", t("heat.teamA")],
    ["team1", t("heat.teamB")],
  ];
  if (currentProfile?.has_ball) {
    options.push(["ball", t("heat.ball")]);
  }
  if (!isLive) {
    (players || []).slice(0, 40).forEach((row) => {
      options.push([`p${row.tracker_id}`, row.name || `#${row.tracker_id}`]);
    });
  }
  return options
    .map(([value, label]) => `<option value="${value}">${esc(label)}</option>`)
    .join("");
}

function buildHeatBlock(container, players, matchId, isLive) {
  const key = container.id;
  const state = heatState[key] || (heatState[key] = { scope: "all" });
  state.matchId = matchId;
  state.live = isLive;

  const old = container.querySelector("img");
  if (old?.dataset.blob) URL.revokeObjectURL(old.dataset.blob);

  container.innerHTML = `
    <div class="sharehead-row">
      <h3>${esc(t("heat.title"))}</h3>
      <select class="lang" data-heat-scope aria-label="${esc(t("heat.title"))}">${heatOptions(players, isLive)}</select>
    </div>
    <p class="hint">${esc(t("heat.hint"))}</p>
    <div class="heatwrap">
      <img alt="${esc(t("heat.title"))}" hidden>
      <p class="muted" hidden>${esc(t("heat.empty"))}</p>
    </div>`;

  const select = container.querySelector("[data-heat-scope]");
  select.value = state.scope;
  if (select.value !== state.scope) {
    state.scope = "all";
    select.value = "all";
  }
  select.addEventListener("change", () => {
    state.scope = select.value;
    loadHeatmap(key);
  });
  loadHeatmap(key);
}

async function loadHeatmap(key) {
  const container = $(key);
  const state = heatState[key];
  if (!container || !state || !state.matchId) return;
  const select = container.querySelector("[data-heat-scope]");
  if (select) state.scope = select.value;

  const bust = state.live ? `&t=${Date.now()}` : "";
  const url = `/api/matches/${state.matchId}/heatmap.png?scope=${encodeURIComponent(state.scope)}${bust}`;
  const image = container.querySelector("img");
  const hint = container.querySelector(".muted");

  try {
    const response = await fetch(url);
    if (response.status === 401) {
      location.href = "/login.html";
      return;
    }
    if (!response.ok) throw new Error(`heatmap ${response.status}`);
    const blob = await response.blob();
    if (image.dataset.blob) URL.revokeObjectURL(image.dataset.blob);
    image.dataset.blob = URL.createObjectURL(blob);
    image.src = image.dataset.blob;
    image.hidden = false;
    hint.hidden = true;
  } catch {
    image.hidden = true;
    hint.hidden = false;
  }
}

/* ------------------------------------------------------------ gol / asist */
function scoreOf(events, profile = currentProfile) {
  const score = { a: 0, b: 0 };
  const points = profile?.points || { goal: 1 };
  (events || []).forEach((event) => {
    const value = Number(points[event.kind] || 0);
    if (!value) return;
    const team = Number(event.team) === 1 ? 1 : 0;
    if ((profile?.error_kinds || []).includes(event.kind)) score[team ? "a" : "b"] += value;
    else score[team ? "b" : "a"] += value;
  });
  return score;
}

async function refreshEvents(boxId, matchId, players, isLive, sport = groupSport) {
  let events = [];
  let sets = null;
  try {
    const payload = await api(`/api/matches/${matchId}/events`);
    events = payload.events || [];
    sets = payload.sets || null;
  } catch {
    events = [];
  }
  renderEvents(boxId, matchId, players, isLive, events, sport, sets);
}

function renderEvents(boxId, matchId, players, isLive, events, sport = groupSport, setState = null) {
  const box = $(boxId);
  if (!box) return;
  const profile = profileOf(sport);
  const keep = box.querySelector("[data-event-player]")?.value || "";
  const score = scoreOf(events, profile);
  const sets = profile.scoring === "sets" ? setState : null;
  const scoreText = sets
    ? sets.complete
      ? t("events.finalSets", { a: sets.wins[0], b: sets.wins[1] })
      : t("events.setScore", {
          set: sets.number,
          a: sets.current[0],
          b: sets.current[1],
        })
    : t("events.score", { a: score.a, b: score.b });
  const setLine = sets && !sets.complete
    ? ` · ${t("events.setsScore", { a: sets.wins[0], b: sets.wins[1] })}`
    : "";
  const emptyKey =
    profile.key === "volleyball" ? "events.emptyVolleyball" : "events.empty";
  const hintKey =
    profile.key === "volleyball"
      ? isLive
        ? "events.liveHintVolleyball"
        : "events.hintVolleyball"
      : isLive
        ? "events.liveHint"
        : "events.hint";
  const rows = events.length
    ? events
        .map(
          (event) => `<li>
        <span class="minute">${Math.round(Number(event.minute) || 0)}'</span>
        <span class="who">${esc(event.name || `#${event.tracker_id}`)}</span>
        <span class="kind ${event.kind}">${esc(t(`events.${event.kind}`))}</span>
        <button type="button" class="ghost" data-event-del="${event.id}">×</button>
      </li>`
        )
        .join("")
    : `<li class="muted">${esc(t(emptyKey))}</li>`;

  box.innerHTML = `
    <div class="sharehead-row">
      <h3>${esc(
        t(
          profile.key === "basketball"
            ? "events.titleBasketball"
            : profile.key === "volleyball"
              ? "events.titleVolleyball"
              : "events.title"
        )
      )}</h3>
      <b class="score">${esc(scoreText)}</b>
    </div>
    <p class="hint">${esc(t(hintKey) + setLine)}</p>
    <ul class="eventlist">${rows}</ul>
    <form class="eventform" data-event-form>
      <select data-event-player aria-label="${esc(t("col.player"))}">
        ${players
          .map((row) => `<option value="${row.tracker_id}">${esc(playerName(row))}</option>`)
          .join("")}
      </select>
      <select data-event-kind aria-label="${esc(t("events.kind"))}">
        ${(profile.events || ["goal", "assist"])
          .map((kind) => `<option value="${kind}">${esc(t(`events.${kind}`))}</option>`)
          .join("")}
      </select>
      ${
        isLive
          ? ""
          : `<input type="number" min="0" max="200" step="1" data-event-minute placeholder="${esc(t("events.minute"))}">`
      }
      <button type="submit" ${players.length ? "" : "disabled"}>${esc(t("events.add"))}</button>
    </form>`;

  const playerSelect = box.querySelector("[data-event-player]");
  if (keep && [...playerSelect.options].some((option) => option.value === keep)) {
    playerSelect.value = keep;
  }

  box.querySelector("[data-event-form]").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      tracker_id: Number(playerSelect.value),
      kind: box.querySelector("[data-event-kind]").value,
    };
    const minuteInput = box.querySelector("[data-event-minute]");
    if (minuteInput && minuteInput.value !== "") body.minute = Number(minuteInput.value);
    try {
      await api(`/api/matches/${matchId}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      toast(t("events.saved"));
      await refreshEvents(boxId, matchId, players, isLive, sport);
    } catch (error) {
      showError(error.message);
    }
  });

  box.querySelectorAll("[data-event-del]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/api/matches/${matchId}/events/${button.dataset.eventDel}`, {
          method: "DELETE",
        });
        toast(t("events.deleted"));
        await refreshEvents(boxId, matchId, players, isLive, sport);
      } catch (error) {
        showError(error.message);
      }
    });
  });
}

/* -------------------------------------------------------- kimlik düzeltme */
function renderRemap(players) {
  const source = $("remapA");
  const target = $("remapB");
  if (!source || !target) return;
  const options = players
    .map(
      (row) =>
        `<option value="${row.tracker_id}">${esc(playerName(row))} · #${row.tracker_id} · ${row.distance_m} m</option>`
    )
    .join("");
  const keepSource = source.value;
  const keepTarget = target.value;
  source.innerHTML = options;
  target.innerHTML = options;
  if ([...source.options].some((option) => option.value === keepSource)) {
    source.value = keepSource;
  }
  if ([...target.options].some((option) => option.value === keepTarget)) {
    target.value = keepTarget;
  } else if (target.options.length > 1) {
    target.selectedIndex = 1;
  }
  $("remapBtn").disabled = players.length < 2;
}

async function applyRemap() {
  if (!lastDetail) return;
  const source = Number($("remapA").value);
  const target = Number($("remapB").value);
  if (source === target) {
    showError(t("remap.same"));
    return;
  }
  if (!confirm(t("remap.confirm"))) return;
  $("remapBtn").disabled = true;
  try {
    await api(`/api/matches/${lastDetail.id}/remap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, target, mode: $("remapMode").value }),
    });
    const fresh = await api(`/api/matches/${lastDetail.id}`);
    cardVersion += 1;
    await renderDetail(fresh, false);
    await loadRoster();
    toast(t("remap.done"));
  } catch (error) {
    showError(error.message);
  } finally {
    $("remapBtn").disabled = false;
  }
}

/* ------------------------------------------------------------------ kartlar */
function cardUrl(match, trackerId) {
  const version = cardVersion ? `&v=${cardVersion}` : "";
  return `/api/matches/${match.id}/players/${trackerId}/card.png?lang=${cardLang()}${version}`;
}

function playerName(row) {
  return row.name || `#${row.tracker_id}`;
}

function renderShareCards(match, players) {
  const grid = $("shareGrid");
  const head = document.querySelector(".sharehead");
  if (!match.analysis_path || !players.length) {
    grid.innerHTML = "";
    if (head) head.hidden = true;
    return;
  }
  if (head) head.hidden = false;

  grid.innerHTML = players
    .map((row) => {
      const url = cardUrl(match, row.tracker_id);
      return `<figure class="sharecard">
        <img loading="lazy" src="${url}" alt="${esc(playerName(row))}" width="1080" height="1350">
        <figcaption>
          <b>${esc(playerName(row))}</b>
          <span>${row.distance_m} m · ${row.top_speed_kmh} km/s</span>
        </figcaption>
        <div class="sharebtns">
          <button type="button" data-act="whatsapp" data-tid="${row.tracker_id}">${esc(t("cards.whatsapp"))}</button>
          <button type="button" data-act="instagram" data-tid="${row.tracker_id}">${esc(t("cards.instagram"))}</button>
          <a class="ghost" href="${url}&download=1" download data-tid="${row.tracker_id}">${esc(t("cards.download"))}</a>
        </div>
      </figure>`;
    })
    .join("");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function shareCard(match, row, platform) {
  const url = cardUrl(match, row.tracker_id);
  const name = playerName(row);
  // Açıklama kartın dilini izler; arayüz dili farklı olabilir.
  const metres = Number(row.distance_m || 0);
  const caption = tIn(cardLang(), "cards.caption", {
    name,
    match: match.title || "",
    distance: metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`,
    speed: row.top_speed_kmh,
  });
  const filename = `bakiappi-${match.id}-${row.tracker_id}.png`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`card ${response.status}`);
    const blob = await response.blob();

    if (typeof File === "function" && navigator.canShare) {
      const file = new File([blob], filename, { type: "image/png" });
      if (navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], text: caption });
        return;
      }
    }

    // Tarayıcı dosya paylaşımını desteklemiyor: kartı indir, metni hazırla.
    downloadBlob(blob, filename);
    if (platform === "whatsapp") {
      window.open(`https://wa.me/?text=${encodeURIComponent(caption)}`, "_blank", "noopener");
      toast(t("cards.downloaded"));
    } else {
      try {
        await navigator.clipboard.writeText(caption);
        toast(t("cards.copied"));
      } catch {
        toast(t("cards.downloaded"));
      }
    }
  } catch (error) {
    if (error && error.name === "AbortError") return;
    const link = document.createElement("a");
    link.href = `${url}&download=1`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    toast(t("cards.shareFailed"));
  }
}

/* -------------------------------------------------------------------- canlı */
function renderLive(match) {
  lastLive = match;
  const stats = match.stats || {};
  const players = stats.players || [];
  const sport = match.sport || groupSport;
  const profile = profileOf(sport);
  const periodsTotal = Number(stats.periods_total || profile.periods || 2);
  const teamA = teamOf(stats, 0);
  const teamB = teamOf(stats, 1);
  const fastest = players.reduce((best, row) => Math.max(best, row.top_speed_kmh), 0);

  $("livePanel").hidden = false;
  const paused = Boolean(stats.paused);
  const period = Number(match.sets?.number || stats.period) || 1;
  $("liveTitle").textContent = match.title || t("app.liveTitle");
  $("liveStage").textContent = paused
    ? t("control.paused")
    : match.stage || t(STATUS_KEY[match.status] || "status.live");
  $("liveClock").textContent = `${clock(stats.elapsed_s)} · ${periodLabel(period, sport)} · ${stats.samples || 0}`;
  $("liveDot").className = `dot ${match.status === "error" ? "bad" : paused ? "paused" : "on"}`;

  const running = match.status === "live" || match.status === "connecting";
  $("stopBtn").disabled = !running;
  $("stopBtn").textContent = running ? t("app.stopBtn") : t("app.finishedBtn");
  $("pauseBtn").disabled = !running;
  $("pauseBtn").textContent = paused ? t("control.resume") : t("control.pause");
  // Voleybolda set, olaylardan hesaplanır; elle "sonraki set" düğmesi
  // set skoruyla çelişmemesi için gösterilmez.
  $("halfBtn").hidden = sport === "volleyball";
  $("halfBtn").disabled = !running || period >= periodsTotal;
  $("halfBtn").textContent = t(
    sport === "basketball"
      ? "control.nextQuarter"
      : sport === "volleyball"
        ? "control.nextSet"
        : "control.half"
  );
  $("calibBtn").hidden = profile.calibration !== "manual";
  const needsCalibration =
    profile.calibration === "manual" &&
    !(match.calibration && match.calibration.points) &&
    stats.calibration !== "manual";
  $("calibBtn").classList.toggle("warn", needsCalibration);

  $("liveCards").innerHTML = [
    card(`${teamA.players} / ${teamB.players}`, t("stat.squad")),
    card(`${teamA.distance_m} m`, t("stat.teamADistance"), "t0"),
    card(`${teamB.distance_m} m`, t("stat.teamBDistance"), "t1"),
    card(`${fastest.toFixed(1)} km/s`, t("stat.topSpeed")),
    card(stats.noise_cm ?? 0, t("stat.noise")),
    stats.ball
      ? card(`%${Math.round((stats.ball.visible_ratio || 0) * 100)}`, t("stat.ball"))
      : "",
  ].join("");

  $("liveRows").innerHTML = playerRows(players);

  const playerKey = players.map((row) => row.tracker_id).join(",");
  if (liveEventsMatch !== match.id || livePlayersKey !== playerKey) {
    liveEventsMatch = match.id;
    livePlayersKey = playerKey;
    refreshEvents("liveEvents", match.id, players, true, sport);
  }

  if (liveHeatId !== match.id) {
    liveHeatId = match.id;
    buildHeatBlock($("liveHeat"), [], match.id, true);
  }
}

function openLive(matchId) {
  liveId = matchId;
  liveEventsMatch = null;
  livePlayersKey = "";
  $("livePanel").hidden = false;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!liveId) return;
    try {
      const match = await api(`/api/matches/${liveId}`);
      renderLive(match);
      liveTick += 1;
      if (liveTick % 5 === 1) loadHeatmap("liveHeat");
      if (match.status === "finished" || match.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        liveId = null;
        $("liveDot").className = "dot";
        if (match.status === "error") showError(match.error || t("msg.streamLost"));
        await loadMatches();
      }
    } catch (error) {
      showError(error.message);
    }
  }, 2000);
}

async function controlLive(payload) {
  if (!liveId) return;
  $("pauseBtn").disabled = true;
  $("halfBtn").disabled = true;
  try {
    const match = await api(`/api/matches/${liveId}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderLive(match);
  } catch (error) {
    showError(error.message);
    if (lastLive) renderLive(lastLive);
  }
}

async function stopLive() {
  if (!liveId) return;
  $("stopBtn").disabled = true;
  try {
    await api(`/api/matches/${liveId}/stop`, { method: "POST" });
    $("liveStage").textContent = t("msg.analysisPreparing");
  } catch (error) {
    showError(error.message);
    $("stopBtn").disabled = false;
  }
}

async function shareMatch() {
  if (!lastDetail) return;
  try {
    const payload = await api(`/api/matches/${lastDetail.id}/share`, { method: "POST" });
    lastDetail.share_token = payload.token;
    const url = `${location.origin}${payload.path}`;
    $("shareLink").href = payload.path;
    $("shareLink").textContent = url;
    $("shareLinkRow").hidden = false;
    try {
      await navigator.clipboard.writeText(url);
      toast(t("share.copied"));
    } catch {
      toast(url);
    }
  } catch (error) {
    showError(error.message);
  }
}

async function loadReport(match) {
  try {
    const response = await fetch(
      `/api/matches/${match.id}/report.txt?lang=${currentLang()}`,
      { credentials: "same-origin" }
    );
    if (!response.ok) throw new Error("report");
    $("detailReport").textContent = await response.text();
  } catch {
    $("detailReport").textContent = "";
  }
}

function applyTrendMetric(value) {
  trendMetric = TREND_METRICS.includes(value) ? value : TREND_METRICS[0];
  $("trendMetric").value = trendMetric;
  saveTrendMetric(trendMetric);
  drawTrend();
}

async function loadTrends() {
  try {
    trendData = await api("/api/roster/trends?limit=12");
  } catch {
    trendData = null;
  }
  drawTrend();
}

function drawTrend() {
  const drawn = TrendChart.render($("trendChart"), $("trendLegend"), {
    matches: trendData?.matches || [],
    series: trendData?.series || [],
    valueOf: (point) => point[trendMetric],
    unit: t(`trend.unit.${trendMetric}`),
    wholeNumbers: trendMetric === "sprints",
  });
  $("trendPanel").hidden = !drawn;
}

async function loadRoster() {
  try {
    const payload = await api("/api/roster");
    const players = payload.players || [];
    const options = $("rosterNames");
    options.innerHTML = (payload.names || [])
      .map((name) => `<option value="${esc(name)}"></option>`)
      .join("");

    $("rosterPanel").hidden = !players.length;
    $("rosterRows").innerHTML = players
      .map(
        (row) => `<tr>
          <td>${esc(row.name)}
            <button type="button" class="ghost" data-rename="${row.player_id || ""}"
                    data-name="${esc(row.name)}" title="${esc(t("roster.rename"))}">✎</button></td>
          <td>${row.matches}</td>
          <td>${row.distance_m} m</td>
          <td>${row.top_speed_kmh} km/s</td>
          <td>${row.sprints || 0}</td>
        </tr>`
      )
      .join("");
    if (players.length) await loadTrends();
  } catch {
    $("rosterPanel").hidden = true;
  }
}

/* ------------------------------------------------------------------ analiz */
async function renderDetail(match, scroll = true) {
  lastDetail = match;
  const stats = match.stats || {};
  const sport = match.sport || groupSport;
  const profile = profileOf(sport);
  const period = Number(stats.period) || (sport === "volleyball" ? 1 : Number(profile.periods) || 2);
  const names = await loadNames(match);
  const players = (stats.players || []).map((row) => ({
    ...row,
    name: names[row.tracker_id],
  }));
  lastPlayers = players;

  const teamA = teamOf(stats, 0);
  const teamB = teamOf(stats, 1);

  $("detailPanel").hidden = false;
  $("detailTitle").textContent = match.title || t("app.analysis");
  $("detailMeta").textContent = `${dateText(match.started_at)} · ${clock(stats.elapsed_s)} · ${periodLabel(
    period,
    sport
  )} · ${players.length} ${t("app.players")}`;
  $("detailCalibBtn").hidden = profile.calibration !== "manual";
  $("detailCalibBtn").classList.toggle(
    "warn",
    profile.calibration === "manual" && !(match.calibration && match.calibration.points)
  );
  $("download").href = `/api/matches/${match.id}/analysis/file`;
  $("download").hidden = !match.analysis_path;
  $("reportLink").href = `/api/matches/${match.id}/report.png?lang=${cardLang()}`;
  $("reportLink").hidden = !match.analysis_path;
  $("csvLink").href = `/api/matches/${match.id}/export.csv`;
  $("csvLink").hidden = !match.analysis_path;
  $("videoLink").hidden = !match.video;
  $("videoLink").href = `/api/matches/${match.id}/video`;
  $("clipRow").hidden = !match.video;
  $("videoBlock").hidden = !match.video;
  if (match.video) {
    $("videoPlayer").src = `/api/matches/${match.id}/video`;
  } else {
    $("videoPlayer").removeAttribute("src");
  }
  $("reportTxt").href = `/api/matches/${match.id}/report.txt?lang=${currentLang()}`;
  $("reportTxt").hidden = !match.analysis_path;
  $("detailReport").textContent = "";
  if (match.analysis_path) loadReport(match);
  $("shareBtn").hidden = !match.analysis_path;
  $("reconnectBtn").hidden = match.status !== "error";

  const shareRow = $("shareLinkRow");
  if (match.share_token) {
    $("shareLink").href = `/s/${match.share_token}`;
    $("shareLink").textContent = `${location.origin}/s/${match.share_token}`;
    shareRow.hidden = false;
  } else {
    shareRow.hidden = true;
  }

  $("detailCards").innerHTML = [
    card(`${teamA.players} / ${teamB.players}`, t("stat.squad")),
    card(`${teamA.distance_m} m`, t("stat.teamADistance"), "t0"),
    card(`${teamB.distance_m} m`, t("stat.teamBDistance"), "t1"),
    card(`${Math.max(teamA.top_speed_kmh, teamB.top_speed_kmh)} km/s`, t("stat.topSpeed")),
    stats.ball
      ? card(`%${Math.round((stats.ball.visible_ratio || 0) * 100)}`, t("stat.ball"))
      : "",
  ].join("");

  $("detailRows").innerHTML = playerRows(players);
  buildHeatBlock($("detailHeat"), players, match.id, false);
  refreshEvents("detailEvents", match.id, players, false, sport);
  renderRemap(players);
  renderNames(players);
  renderShareCards(match, players);
  if (scroll) $("detailPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadMatches() {
  const box = $("matchList");
  try {
    const matches = await api("/api/matches");
    if (!matches.length) {
      box.innerHTML = `<span class="muted">${esc(t("msg.noMatches"))}</span>`;
      return;
    }
    box.innerHTML = "";
    matches.forEach((match) => {
      const stats = match.stats || {};
      const row = document.createElement("button");
      row.className = "matchrow";
      row.innerHTML = `<b>${esc(match.title)}</b>
        <span class="pill sport">${esc(sportName(match.sport || groupSport))}</span>
        <span class="pill ${match.status}">${esc(t(STATUS_KEY[match.status] || "status.live"))}</span>
        <span class="muted">${dateText(match.started_at)}</span>
        <span class="muted">${clock(stats.elapsed_s)}</span>
        <span class="muted">${(stats.players || []).length} ${esc(t("app.players"))}</span>`;
      row.onclick = () => {
        if (match.status === "live" || match.status === "connecting") openLive(match.id);
        else renderDetail(match);
      };
      box.appendChild(row);
    });
  } catch (error) {
    box.innerHTML = `<span class="muted">${esc(t("msg.matchesFailed"))}</span>`;
    showError(error.message);
  }
}

$("startForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("startErr").hidden = true;
  $("startBtn").disabled = true;
  try {
    const match = await api("/api/matches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: $("matchTitle").value,
        source_url: $("sourceUrl").value,
        sport: groupSport,
      }),
    });
    $("sourceUrl").value = "";
    $("matchTitle").value = "";
    openLive(match.id);
    await loadMatches();
  } catch (error) {
    $("startErr").textContent = serverMessage(error.message);
    $("startErr").hidden = false;
  } finally {
    $("startBtn").disabled = false;
  }
});

$("stopBtn").addEventListener("click", stopLive);
$("pauseBtn").addEventListener("click", () =>
  controlLive({ paused: !(lastLive?.stats?.paused) })
);
$("halfBtn").addEventListener("click", () => {
  const current = Number(lastLive?.stats?.period) || 1;
  controlLive({ period: current + 1 });
});
$("sportChange").addEventListener("click", () => openSportGate(false));
$("sportContinue").addEventListener("click", saveSport);
$("calibBtn").addEventListener("click", () => {
  if (lastLive) openCalibration(lastLive);
});
$("detailCalibBtn").addEventListener("click", () => {
  if (lastDetail) openCalibration(lastDetail);
});
$("calibClose").addEventListener("click", () => {
  $("calibModal").hidden = true;
});
$("calibReset").addEventListener("click", () => {
  calib.corners = [];
  drawCalibration();
});
$("calibSave").addEventListener("click", saveCalibration);
$("calibExtent").addEventListener("change", () => {
  calib.extent = $("calibExtent").value === "half" ? "half" : "full";
  calib.corners = [];
  drawCalibration();
});
$("calibCanvas").addEventListener("click", (event) => {
  if (!calib.image || calib.corners.length >= 4) return;
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  calib.corners.push([Math.round(x), Math.round(y)]);
  drawCalibration();
});
$("saveNames").addEventListener("click", saveNames);
$("remapBtn").addEventListener("click", applyRemap);
$("shareBtn").addEventListener("click", shareMatch);
$("deleteBtn").addEventListener("click", deleteMatch);
$("reconnectBtn").addEventListener("click", reconnectMatch);

async function deleteMatch() {
  if (!lastDetail) return;
  if (!confirm(t("match.deleteConfirm"))) return;
  try {
    await api(`/api/matches/${lastDetail.id}`, { method: "DELETE" });
    $("detailPanel").hidden = true;
    lastDetail = null;
    lastPlayers = [];
    await loadMatches();
    await loadRoster();
    toast(t("match.deleted"));
  } catch (error) {
    showError(error.message);
  }
}

async function reconnectMatch() {
  if (!lastDetail) return;
  try {
    await api(`/api/matches/${lastDetail.id}/reconnect`, { method: "POST" });
    await loadMatches();
    openLive(lastDetail.id);
    toast(t("match.reconnecting"));
  } catch (error) {
    showError(error.message);
  }
}

$("rosterRows").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-rename]");
  if (!button || !button.dataset.rename) return;
  const next = prompt(t("roster.renamePrompt"), button.dataset.name);
  if (!next || !next.trim() || next.trim() === button.dataset.name) return;
  try {
    await api(`/api/players/${button.dataset.rename}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: next.trim() }),
    });
    await loadRoster();
    if (lastDetail) await renderDetail(lastDetail, false);
    toast(t("roster.renamed"));
  } catch (error) {
    showError(error.message);
  }
});

$("nameGrid").addEventListener("input", (event) => {
  const input = event.target.closest("input[data-tid]");
  if (input) draftNames.set(Number(input.dataset.tid), input.value);
});

$("shareGrid").addEventListener("click", (event) => {
  const target = event.target.closest("[data-act]");
  if (!target || !lastDetail) return;
  const row = lastPlayers.find((player) => String(player.tracker_id) === target.dataset.tid);
  if (!row) return;
  shareCard(lastDetail, row, target.dataset.act);
});

$("trendMetric").addEventListener("change", (event) => {
  applyTrendMetric(event.target.value);
});
$("trendMetric").value = trendMetric;

$("clipBtn").addEventListener("click", () => {
  if (!lastDetail || !lastDetail.video) return;
  const start = Math.max(0, Number($("clipStart").value) || 0);
  const length = Math.min(180, Math.max(1, Number($("clipLength").value) || 20));
  window.open(
    `/api/matches/${lastDetail.id}/video/clip?start=${start}&duration=${length}`,
    "_blank",
    "noopener"
  );
});

$("logout").addEventListener("click", async () => {
  await fetch("/api/auth/logout", { method: "POST" });
  location.href = "/login.html";
});

document.addEventListener("i18n:changed", () => {
  $("err").hidden = true;
  liveHeatId = null;
  setSportUI();
  if ($("sportGate") && !$("sportGate").hidden) renderSportChoices();
  if ($("calibModal") && !$("calibModal").hidden) drawCalibration();
  loadMatches();
  if (lastLive) renderLive(lastLive);
  if (lastDetail) renderDetail(lastDetail, false);
  // Eksen birimi ve tarih biçimi dile bağlı; grafiği aynı veriyle yeniden çiz.
  drawTrend();
});

document.addEventListener("cardlang:changed", () => {
  if (lastDetail) renderShareCards(lastDetail, lastPlayers);
});

/* ------------------------------------------------------------ branş kapısı */
function setSportUI() {
  currentProfile = profileOf(groupSport);
  const tag = $("sportTag");
  if (tag) tag.textContent = sportName(groupSport);
  document.body.dataset.sport = groupSport;
  const title = $("matchTitle");
  if (title) {
    title.placeholder = t(
      {
        basketball: "app.titlePlaceholderBasketball",
        volleyball: "app.titlePlaceholderVolleyball",
      }[groupSport] || "app.titlePlaceholder"
    );
  }
}

function renderSportChoices() {
  const box = $("sportChoices");
  if (!box) return;
  box.innerHTML = profiles
    .map((profile) => {
      const on = profile.key === groupSport ? " on" : "";
      const desc = t(
        {
          basketball: "sport.basketballDesc",
          volleyball: "sport.volleyballDesc",
        }[profile.key] || "sport.footballDesc"
      );
      const method = t(profile.calibration === "manual" ? "sport.manual" : "sport.auto");
      return `<button type="button" class="sportcard${on}" data-sport="${profile.key}">
        <span class="emoji">${profile.emoji || ""}</span>
        <b>${esc(sportName(profile.key))}</b>
        <span>${esc(desc)}</span>
        <em>${esc(t("sport.court"))}: ${profile.court.length_m} × ${profile.court.width_m} m · ${profile.periods} ${esc(
          t("sport.periods").toLowerCase()
        )} · ${esc(method)}</em>
      </button>`;
    })
    .join("");
  box.querySelectorAll("[data-sport]").forEach((button) => {
    button.addEventListener("click", () => {
      groupSport = button.dataset.sport;
      currentProfile = profileOf(groupSport);
      renderSportChoices();
    });
  });
  $("sportContinue").disabled = profiles.length === 0;
}

function sportGateKey() {
  return `bakiappi.sportGate.${$("groupTag").textContent || "grup"}`;
}

function openSportGate() {
  renderSportChoices();
  $("sportGate").hidden = false;
  $("sportContinue").disabled = false;
}

async function saveSport() {
  $("sportContinue").disabled = true;
  try {
    const payload = await api("/api/sport", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sport: groupSport }),
    });
    groupSport = payload.sport || groupSport;
    try {
      localStorage.setItem(sportGateKey(), groupSport);
    } catch {
      /* özel pencere: kapı bu oturumda kalır */
    }
    $("sportGate").hidden = true;
    setSportUI();
    clearInterval(pollTimer);
    pollTimer = null;
    liveId = null;
    liveEventsMatch = null;
    livePlayersKey = "";
    lastDetail = null;
    lastLive = null;
    liveHeatId = null;
    $("detailPanel").hidden = true;
    $("livePanel").hidden = true;
    await loadMatches();
    await loadRoster();
    toast(t("sport.saved"));
  } catch (error) {
    showError(error.message);
  } finally {
    $("sportContinue").disabled = false;
  }
}

/* ------------------------------------------------------------ kalibrasyon */
function drawCalibration() {
  const canvas = $("calibCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = calib.image ? calib.image.naturalWidth : 1280;
  const height = calib.image ? calib.image.naturalHeight : 720;
  canvas.width = width;
  canvas.height = height;
  ctx.fillStyle = "#060a0f";
  ctx.fillRect(0, 0, width, height);
  if (calib.image) ctx.drawImage(calib.image, 0, 0, width, height);

  const radius = Math.max(7, width / 90);
  const line = Math.max(2, width / 500);
  ctx.lineWidth = line;
  ctx.strokeStyle = "#FF1493";
  ctx.fillStyle = "#FF1493";
  if (calib.corners.length > 1) {
    ctx.beginPath();
    calib.corners.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (calib.corners.length === 4) ctx.closePath();
    ctx.stroke();
  }
  calib.corners.forEach(([x, y], index) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#080b10";
    ctx.font = `bold ${Math.round(radius * 1.2)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), x, y + 1);
    ctx.fillStyle = "#FF1493";
  });
  $("calibPoints").textContent = t("calib.points", { n: calib.corners.length });
  $("calibSave").disabled = calib.corners.length !== 4;
}

function openCalibration(match) {
  calib.match = match;
  calib.corners = [];
  calib.image = null;
  calib.extent = match.calibration?.extent === "half" ? "half" : "full";
  $("calibExtent").value = calib.extent;
  $("calibModal").hidden = false;
  const loading = $("calibLoading");
  loading.hidden = false;
  loading.textContent = t("common.loading");
  drawCalibration();

  const image = new Image();
  image.onload = () => {
    calib.image = image;
    loading.hidden = true;
    drawCalibration();
  };
  image.onerror = () => {
    loading.textContent = t("calib.frameFailed");
    showError(t("calib.frameFailed"));
  };
  image.src = `/api/matches/${match.id}/frame.jpg?t=1&v=${Date.now()}`;
}

async function saveCalibration() {
  if (!calib.match || calib.corners.length !== 4) return;
  $("calibSave").disabled = true;
  try {
    await api(`/api/matches/${calib.match.id}/calibration`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: calib.corners, extent: calib.extent }),
    });
    const calibration = { points: calib.corners, extent: calib.extent };
    const updated = { ...calib.match, calibration };
    $("calibModal").hidden = true;
    if (lastLive && lastLive.id === calib.match.id) lastLive.calibration = "manual";
    if (lastDetail && lastDetail.id === calib.match.id) await renderDetail(updated, false);
    toast(t("calib.saved"));
  } catch (error) {
    showError(error.message);
    $("calibSave").disabled = false;
  }
}

async function boot() {
  try {
    const [group, sportsPayload] = await Promise.all([api("/api/auth/me"), api("/api/sports")]);
    groupSport = group.sport || "football";
    profiles = sportsPayload.sports || [];
    $("groupTag").textContent = group.name;
    setSportUI();
  } catch {
    return;
  }
  let acknowledged = null;
  try {
    acknowledged = localStorage.getItem(sportGateKey());
  } catch {
    acknowledged = null;
  }
  if (acknowledged !== groupSport) openSportGate();
  await loadMatches();
  await loadRoster();
}

boot();
