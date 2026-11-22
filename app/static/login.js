const $ = (id) => document.getElementById(id);

function showError(message) {
  const box = $("err");
  box.textContent = serverMessage(String(message));
  box.hidden = false;
}

async function send(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || response.statusText);
  return payload;
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((node) => node.classList.remove("on"));
    button.classList.add("on");
    $("err").hidden = true;
    $("loginForm").hidden = button.dataset.tab !== "login";
    $("registerForm").hidden = button.dataset.tab !== "register";
  });
});

$("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("err").hidden = true;
  try {
    await send("/api/auth/login", {
      username: $("loginUser").value,
      password: $("loginPass").value,
    });
    location.href = "/app.html";
  } catch (error) {
    showError(error.message);
  }
});

$("registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("err").hidden = true;
  if ($("regPass").value !== $("regPass2").value) {
    showError(t("err.passwordMismatch"));
    return;
  }
  try {
    await send("/api/auth/register", {
      group_name: $("regGroup").value,
      username: $("regUser").value,
      password: $("regPass").value,
    });
    location.href = "/app.html";
  } catch (error) {
    showError(error.message);
  }
});

document.addEventListener("i18n:changed", () => {
  $("err").hidden = true;
});

fetch("/api/auth/me")
  .then((response) => {
    if (response.ok) location.href = "/app.html";
  })
  .catch(() => {});
