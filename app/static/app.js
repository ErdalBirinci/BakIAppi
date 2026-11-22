const $ = (id) => document.getElementById(id);

let selection = { type: "none", name: null, file: null };
let pollTimer = null;
let sport = "football";
const calib = { corners: [], frame: null, url: null, extent: "full" };

function showError(message) {
  const box = $("err");
  box.textContent = String(message);
  box.hidden = false;
}

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)} sn`;
  return `${Math.floor(seconds / 60)} dk ${Math.round(seconds % 60)} sn`;
}

function needsManualCalibration(value = sport) {
  return value === "basketball" || value === "volleyball";
}

function updateRun() {
  const ready = selection.type !== "none";
  const needsCalibration = needsManualCalibration() && calib.corners.length !== 4;
  $("run").disabled = !ready || needsCalibration;
}

function selectSample(chip, sample) {
  selection = { type: "sample", name: sample.name, file: null };
  document.querySelectorAll(".chip[data-sample]").forEach((node) => node.classList.remove("on"));
  chip.classList.add("on");
  $("fileLabel").textContent = "veya dosya seç";
  updateRun();
  if (needsManualCalibration()) prepareFrame();
}

function sportChanged() {
  const manual = needsManualCalibration();
  $("calibPanel").hidden = !manual;
  document.querySelectorAll("[data-sport]").forEach((button) => {
    button.classList.toggle("on", button.dataset.sport === sport);
  });
  updateRun();
  if (manual) prepareFrame();
}

async function loadSamples() {
  const box = $("samples");
  try {
    const samples = await fetch("/api/samples").then((response) => response.json());
    box.innerHTML = "";
    if (!samples.length) {
      box.innerHTML = '<span class="muted">örnek video yok — dosya yükle</span>';
      return;
    }
    samples.forEach((sample, index) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.dataset.sample = sample.name;
      chip.textContent = `${sample.name} · ${sample.size_mb} MB`;
      chip.onclick = () => selectSample(chip, sample);
      box.appendChild(chip);
      if (index === 0) chip.click();
    });
  } catch (error) {
    box.innerHTML = '<span class="muted">örnekler yüklenemedi</span>';
    showError(error.message);
  }
}

/* --------------------------------------------------- basketbol kalibrasyonu */
function drawCalibration() {
  const canvas = $("calibCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = calib.frame ? calib.frame.videoWidth || 1280 : 1280;
  const height = calib.frame ? calib.frame.videoHeight || 720 : 720;
  canvas.width = width;
  canvas.height = height;
  ctx.fillStyle = "#060a0f";
  ctx.fillRect(0, 0, width, height);
  if (calib.frame) ctx.drawImage(calib.frame, 0, 0, width, height);

  const radius = Math.max(7, width / 90);
  ctx.lineWidth = Math.max(2, width / 500);
  ctx.strokeStyle = "#00BFFF";
  ctx.fillStyle = "#00BFFF";
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
    ctx.fillStyle = "#00BFFF";
  });
  $("calibHint").textContent = `${calib.corners.length}/4 köşe işaretlendi`;
  updateRun();
}

function prepareFrame() {
  const video = $("calibVideo");
  if (!video || selection.type === "none") return;
  if (calib.url) URL.revokeObjectURL(calib.url);
  calib.url = null;
  calib.corners = [];
  calib.frame = null;
  if (selection.type === "file") {
    calib.url = URL.createObjectURL(selection.file);
    video.src = calib.url;
  } else {
    video.src = `/api/samples/${encodeURIComponent(selection.name)}`;
  }
  video.load();
  video.onloadeddata = () => {
    const target = Math.min(1, Math.max(0, (video.duration || 2) / 2));
    if (video.currentTime && Math.abs(video.currentTime - target) < 0.05) {
      calib.frame = video;
      drawCalibration();
    } else {
      video.currentTime = target;
    }
  };
  video.onseeked = () => {
    calib.frame = video;
    drawCalibration();
  };
  drawCalibration();
}

$("sportRow").addEventListener("click", (event) => {
  const button = event.target.closest("[data-sport]");
  if (!button) return;
  sport = ["football", "basketball", "volleyball"].includes(button.dataset.sport)
    ? button.dataset.sport
    : "football";
  sportChanged();
});

$("calibCanvas").addEventListener("click", (event) => {
  if (!calib.frame || calib.corners.length >= 4) return;
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  calib.corners.push([Math.round(x), Math.round(y)]);
  drawCalibration();
});

$("calibReset").addEventListener("click", () => {
  calib.corners = [];
  drawCalibration();
});

$("calibExtent").addEventListener("change", () => {
  calib.extent = $("calibExtent").value === "half" ? "half" : "full";
  calib.corners = [];
  drawCalibration();
});

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  selection = { type: "file", name: file.name, file };
  document.querySelectorAll(".chip[data-sample]").forEach((node) => node.classList.remove("on"));
  $("fileLabel").textContent = file.name;
  updateRun();
  if (needsManualCalibration()) prepareFrame();
});

$("run").addEventListener("click", async () => {
  $("err").hidden = true;
  $("resultPanel").hidden = true;
  $("progressPanel").hidden = false;
  $("barFill").style.width = "0%";
  $("stage").textContent = "gönderiliyor";
  $("frames").textContent = "";
  $("eta").textContent = "";
  $("run").disabled = true;

  const body = new FormData();
  body.append("max_seconds", $("maxSeconds").value);
  body.append("sport", sport);
  if (needsManualCalibration()) {
    body.append("calibration", JSON.stringify({ points: calib.corners, extent: calib.extent }));
  }
  if (selection.type === "file") body.append("file", selection.file);
  else body.append("sample", selection.name);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    watch(payload.id);
  } catch (error) {
    showError(error.message);
    updateRun();
  }
});

function watch(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try {
      job = await fetch(`/api/jobs/${jobId}`).then((response) => response.json());
    } catch {
      return;
    }

    const percent = Math.max(0, Math.min(100, job.percent || 0));
    $("barFill").style.width = `${percent}%`;
    $("stage").textContent = job.stage || job.status;

    if (job.frames_total) {
      $("frames").textContent = `${job.frames_done} / ${job.frames_total} kare`;
      const remaining =
        percent > 4 && percent < 100 ? (job.elapsed_s / percent) * (100 - percent) : null;
      $("eta").textContent = remaining === null ? "" : `~${formatTime(remaining)} kaldı`;
    }

    if (job.status === "done") {
      clearInterval(pollTimer);
      render(jobId, job.stats);
      updateRun();
    } else if (job.status === "error") {
      clearInterval(pollTimer);
      showError(job.error || "analiz başarısız");
      updateRun();
    }
  }, 1000);
}

function card(value, label, className = "") {
  return `<div class="card ${className}"><b>${value}</b><span>${label}</span></div>`;
}

function teamOf(stats, index) {
  return (stats.teams || {})[String(index)] || { players: 0, distance_m: 0, top_speed_kmh: 0 };
}

function render(jobId, stats, scroll = true) {
  const video = $("out");
  video.src = `/api/jobs/${jobId}/video`;
  video.load();

  const teamA = teamOf(stats, 0);
  const teamB = teamOf(stats, 1);
  const topSpeed = Math.max(teamA.top_speed_kmh, teamB.top_speed_kmh);
  const sportKey = ["football", "basketball", "volleyball"].includes(stats.sport)
    ? stats.sport
    : "football";
  const basketball = sportKey === "basketball";
  const volleyball = sportKey === "volleyball";
  const sportLabel = volleyball ? "Voleybol" : basketball ? "Basketbol" : "Futbol";
  const periods = stats.periods || (basketball ? 4 : volleyball ? 5 : 2);
  const periodLabel = volleyball ? "set" : basketball ? "çeyrek" : "devre";
  const ball = stats.ball || null;

  $("cards").innerHTML = [
    card(sportLabel, "branş", volleyball ? "t1" : basketball ? "t1" : "t0"),
    card(`${periods}`, periodLabel),
    card(`${stats.duration_s} sn`, "işlenen süre"),
    card(stats.players.tracked, "takip edilen oyuncu"),
    card(`${teamA.distance_m} m`, "takım A mesafe", "t0"),
    card(`${teamB.distance_m} m`, "takım B mesafe", "t1"),
    card(`${topSpeed} km/s`, "en yüksek hız"),
    ball ? card(`%${Math.round((ball.visible_ratio || 0) * 100)}`, "top görünürlüğü") : "",
    card(`${stats.elapsed_s} sn`, "işlem süresi"),
  ].join("");

  const rows = stats.player_rows
    .map(
      (row) => `<tr>
        <td>#${row.tracker_id}</td>
        <td class="team${row.team}">${row.team === 0 ? "A" : "B"}</td>
        <td>${row.distance_m} m</td>
        <td>${row.top_speed_kmh} km/s</td>
        <td>${row.avg_speed_kmh} km/s</td>
      </tr>`
    )
    .join("");

  $("rows").innerHTML =
    rows ||
    '<tr><td colspan="5" class="muted">Bu kesitte takip edilebilen oyuncu bulunamadı.</td></tr>';

  $("resultNote").textContent = volleyball
    ? "Mesafe, işaretlediğin dört FIVB kort köşesinden kurulan homografiyle hesaplanır; " +
      "kort 18×9 m, hız bölgeleri 4/8/13 km/s'dir. Setler 25 sayı (5. set 15) ve 2 fark " +
      "kuralıyla hesaplanır; ace, hücum, blok ve hata olaylarını maç panelinden işle."
    : basketball
    ? "Mesafe, işaretlediğin dört kort köşesinden kurulan homografiyle hesaplanır; " +
      "FIBA sahası 28×15 m, hız bölgeleri 6/11/17 km/s'dir. Kamera sabit değilse köşeleri " +
      "yayının başından işaretlemek en doğru sonucu verir."
    : "Mesafe ve hız, saha anahtar noktalarından hesaplanan homografiye dayanır. Tüm " +
      "oyuncuları birlikte kaydıran sistematik bileşen her karede ölçülüp hareketten çıkarılır; " +
      "kalan tespit gürültüsü de karesel olarak düşülür. Yine de değerler yaklaşıktır.";

  $("resultPanel").hidden = false;
  if (scroll) {
    $("resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function restoreLastResult() {
  try {
    const jobs = await fetch("/api/jobs").then((response) => response.json());
    const done = jobs.find((job) => job.status === "done");
    if (!done) return;
    const job = await fetch(`/api/jobs/${done.id}`).then((response) => response.json());
    if (job.stats) render(done.id, job.stats, false);
  } catch {
    /* sonuç geri yüklenemedi; kullanıcı yeni analiz başlatabilir */
  }
}

loadSamples();
sportChanged();
restoreLastResult();
