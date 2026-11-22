/* BakIAppi — tanıtım sayfası etkileşimleri: 3D eğim, kaydırma animasyonu, ilerleme. */
(function () {
  const root = document.documentElement;
  const scene = document.getElementById("scene");
  const nav = document.getElementById("nav");
  const bar = document.getElementById("scrollBar");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* fare ile sahneyi eğ */
  if (scene && finePointer && !reduceMotion) {
    const inner = scene.querySelector(".scene-inner");
    let targetX = 6;
    let targetY = -10;
    let currentX = targetX;
    let currentY = targetY;
    let frame = 0;

    const render = () => {
      frame = 0;
      currentX += (targetX - currentX) * 0.08;
      currentY += (targetY - currentY) * 0.08;
      inner.style.setProperty("--rx", currentX.toFixed(2) + "deg");
      inner.style.setProperty("--ry", currentY.toFixed(2) + "deg");
      if (Math.abs(targetX - currentX) > 0.05 || Math.abs(targetY - currentY) > 0.05) {
        frame = requestAnimationFrame(render);
      }
    };

    const queue = () => { if (!frame) frame = requestAnimationFrame(render); };

    scene.addEventListener("pointermove", (event) => {
      const box = scene.getBoundingClientRect();
      const px = (event.clientX - box.left) / box.width - 0.5;
      const py = (event.clientY - box.top) / box.height - 0.5;
      targetY = -10 + px * 16;
      targetX = 6 - py * 12;
      queue();
    });

    scene.addEventListener("pointerleave", () => {
      targetX = 6;
      targetY = -10;
      queue();
    });
  }

  /* bölümler göründükçe belirsin */
  const items = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("in"));
  } else {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("in");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.12 }
    );
    items.forEach((item, index) => {
      item.style.transitionDelay = Math.min(index % 3, 2) * 70 + "ms";
      observer.observe(item);
    });
  }

  /* nav durumu + okuma ilerlemesi */
  const onScroll = () => {
    const top = window.scrollY;
    if (nav) nav.classList.toggle("scrolled", top > 24);
    if (bar) {
      const height = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = height > 0 ? Math.min(1, top / height) : 0;
      bar.style.width = (ratio * 100).toFixed(2) + "%";
    }
  };

  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);

  /* yumuşak kaydırma (tarayıcı desteği yoksa) */
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const id = link.getAttribute("href").slice(1);
      const target = id && document.getElementById(id);
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "start",
      });
      history.replaceState(null, "", "#" + id);
    });
  });

  /* mobil menü: bağlantı, dış tıklama, Escape, kaydırma ve masaüstüne geçişte kapanır */
  const toggle = document.getElementById("navToggle");
  const menu = document.getElementById("navMenu");
  if (nav && toggle && menu) {
    const isOpen = () => nav.classList.contains("open");
    const close = () => {
      nav.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    };

    toggle.addEventListener("click", () => {
      const open = !isOpen();
      nav.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
    });
    menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", close));
    document.addEventListener("click", (event) => {
      if (isOpen() && !nav.contains(event.target)) close();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !isOpen()) return;
      close();
      toggle.focus();
    });
    window.addEventListener("scroll", () => isOpen() && close(), { passive: true });
    window.matchMedia("(min-width: 881px)").addEventListener("change", close);
  }

  root.classList.add("js");
})();
