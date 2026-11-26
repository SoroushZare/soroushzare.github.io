/* Soroush Zare — site behaviour. Everything here is progressive enhancement:
   the page is complete without it. */
(function () {
  'use strict';

  var root = document.documentElement;
  var darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var themeListeners = [];

  /* ---------- Theme toggle ---------- */
  function currentTheme() {
    return root.getAttribute('data-theme') || (darkQuery.matches ? 'dark' : 'light');
  }
  function notifyTheme() { themeListeners.forEach(function (fn) { fn(); }); }

  var toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
      notifyTheme();
    });
  }
  if (darkQuery.addEventListener) darkQuery.addEventListener('change', notifyTheme);

  /* ---------- Mobile menu closes after a choice ---------- */
  var navCheck = document.getElementById('nav-check');
  document.querySelectorAll('.nav a').forEach(function (a) {
    a.addEventListener('click', function () { if (navCheck) navCheck.checked = false; });
  });

  /* ---------- Publication filter ---------- */
  var filters = document.querySelectorAll('.filter');
  filters.forEach(function (button) {
    button.addEventListener('click', function () {
      var tag = button.getAttribute('data-filter');
      filters.forEach(function (b) {
        var on = b === button;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      document.querySelectorAll('.year').forEach(function (year) {
        var visible = 0;
        year.querySelectorAll('.pub').forEach(function (pub) {
          var show = tag === 'all' || pub.getAttribute('data-tags').split(' ').indexOf(tag) !== -1;
          pub.classList.toggle('is-hidden', !show);
          if (show) visible++;
        });
        year.classList.toggle('is-empty', visible === 0);
      });
    });
  });

  /* ---------- Figure lightbox ---------- */
  if (typeof HTMLDialogElement !== 'undefined') {
    var dialog = document.createElement('dialog');
    dialog.className = 'lightbox';
    var big = document.createElement('img');
    dialog.appendChild(big);
    document.body.appendChild(dialog);
    dialog.addEventListener('click', function () { dialog.close(); });
    document.querySelectorAll('a.zoom').forEach(function (link) {
      link.addEventListener('click', function (event) {
        if (event.metaKey || event.ctrlKey || event.shiftKey) return;
        event.preventDefault();
        var thumb = link.querySelector('img');
        big.src = link.getAttribute('href');
        big.alt = thumb ? thumb.alt : '';
        dialog.showModal();
      });
    });
  }

  /* ---------- Copy email ---------- */
  var copy = document.getElementById('copy-email');
  if (copy && navigator.clipboard) {
    copy.addEventListener('click', function () {
      navigator.clipboard.writeText(copy.getAttribute('data-email')).then(function () {
        var label = copy.textContent;
        copy.textContent = 'Copied';
        setTimeout(function () { copy.textContent = label; }, 1600);
      }).catch(function () {});
    });
  } else if (copy) {
    copy.hidden = true;
  }

  /* ---------- Synthetic EEG panel ----------
     Six channels over the motor cortex. Each is a slow background plus a 10 Hz
     mu rhythm and a little 20 Hz beta. "Imagining" a hand movement lowers
     mu/beta on the opposite hemisphere; the readout is computed from the
     traces themselves (recent power, left vs right), so it lags like a real
     decoder would. Purely illustrative: no real data. */
  var panel = document.getElementById('eeg-panel');
  if (panel) initEEG(panel);

  function initEEG(panel) {
    var canvas = panel.querySelector('canvas');
    var ctx = canvas.getContext && canvas.getContext('2d');
    if (!ctx) return;
    var valueEl = panel.querySelector('.chip__value');
    var barEl = panel.querySelector('.chip__bar span');

    var CHANNELS = ['FC3', 'C3', 'CP3', 'FC4', 'C4', 'CP4']; // first three: left hemisphere
    var RATE = 120;                 // samples per second
    var N = RATE * 4;               // four seconds on screen
    var TWO_PI = Math.PI * 2;

    var seed = 7;
    function rand() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }

    var chans = CHANNELS.map(function () {
      var slow = [];
      for (var i = 0; i < 4; i++) slow.push({ f: 0.6 + rand() * 5, a: 0.18 + rand() * 0.22, p: rand() * TWO_PI });
      return { slow: slow, muF: 9.4 + rand() * 1.6, muP: rand() * TWO_PI, envF: 0.25 + rand() * 0.3, envP: rand() * TWO_PI, betaP: rand() * TWO_PI, buf: new Float32Array(N) };
    });

    var head = 0, t = 0;
    var gain = [1, 1], target = [1, 1];   // mu/beta gain per hemisphere [left, right]
    var intent = 'rest';                  // what is being "imagined"
    var decoded = 'REST', smooth = 0, shown = '';

    function setIntent(next) {
      intent = next;
      // Right hand quiets the LEFT hemisphere, and vice versa.
      target[0] = next === 'right' ? 0.25 : 1;
      target[1] = next === 'left' ? 0.25 : 1;
    }

    function step() {
      var k = 1 / (0.3 * RATE);
      gain[0] += (target[0] - gain[0]) * k * 3;
      gain[1] += (target[1] - gain[1]) * k * 3;
      for (var c = 0; c < chans.length; c++) {
        var ch = chans[c], g = gain[c < 3 ? 0 : 1], v = 0;
        for (var i = 0; i < ch.slow.length; i++) v += ch.slow[i].a * Math.sin(TWO_PI * ch.slow[i].f * t + ch.slow[i].p);
        var env = 0.7 + 0.3 * Math.sin(TWO_PI * ch.envF * t + ch.envP);
        v += g * (env * Math.sin(TWO_PI * ch.muF * t + ch.muP) + 0.3 * Math.sin(TWO_PI * 20 * t + ch.betaP));
        v += (rand() - 0.5) * 0.4;
        ch.buf[head] = v;
      }
      head = (head + 1) % N;
      t += 1 / RATE;
    }

    // Power of the first difference over the last second (emphasises mu/beta).
    function power(from, to) {
      var sum = 0;
      for (var c = from; c < to; c++) {
        var buf = chans[c].buf;
        for (var i = 1; i < RATE; i++) {
          var a = buf[(head - i + N) % N], b = buf[(head - i - 1 + N * 2) % N];
          sum += (a - b) * (a - b);
        }
      }
      return sum;
    }

    function decode() {
      var ratio = Math.log(power(0, 3) / power(3, 6));   // < 0: left hemisphere quieter
      smooth += (ratio - smooth) * 0.08;
      var mag = Math.abs(smooth);
      if (decoded === 'REST') {
        if (mag > 0.8) decoded = smooth < 0 ? 'RIGHT HAND' : 'LEFT HAND';
      } else if (mag < 0.4) {
        decoded = 'REST';
      }
      var confidence = decoded === 'REST' ? Math.max(0.5, 1 - mag) : Math.min(0.99, 0.5 + mag / 3);
      if (decoded !== shown) { shown = decoded; valueEl.textContent = decoded; }
      barEl.style.width = Math.round(confidence * 100) + '%';
    }

    var width = 0, height = 0, colors = {};
    function readColors() {
      var cs = getComputedStyle(root);
      colors.ink = cs.getPropertyValue('--ink').trim();
      colors.muted = cs.getPropertyValue('--ink-3').trim();
      colors.accent = cs.getPropertyValue('--accent').trim();
    }
    function resize() {
      var rect = panel.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width; height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      var top = height * 0.07, rowH = (height * 0.5) / chans.length, left = 46, amp = rowH * 0.2;
      ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      for (var c = 0; c < chans.length; c++) {
        var y0 = top + rowH * (c + 0.5) + (c >= 3 ? rowH * 0.5 : 0);
        var quiet = gain[c < 3 ? 0 : 1] < 0.6;
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = colors.muted;
        ctx.fillText(CHANNELS[c], 12, y0);
        ctx.globalAlpha = quiet ? 0.95 : 0.34;
        ctx.strokeStyle = quiet ? colors.accent : colors.ink;
        ctx.beginPath();
        var buf = chans[c].buf, span = width - left;
        for (var i = 0; i < N; i++) {
          var x = left + (i / (N - 1)) * span, y = y0 - buf[(head + i) % N] * amp;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // --- input: pointer side picks the imagined hand; idle runs an autopilot
    var manualUntil = 0, pointerInside = false;
    function sideOf(event) {
      var rect = panel.getBoundingClientRect();
      return (event.clientX - rect.left) < rect.width / 2 ? 'left' : 'right';
    }
    panel.addEventListener('pointermove', function (e) { pointerInside = true; setIntent(sideOf(e)); });
    panel.addEventListener('pointerdown', function (e) { setIntent(sideOf(e)); manualUntil = performance.now() + 2600; });
    panel.addEventListener('pointerleave', function () { pointerInside = false; manualUntil = performance.now() + 1500; setIntent('rest'); });

    function autopilot(now) {
      if (pointerInside || now < manualUntil) return;
      var phase = (now / 1000) % 13;
      setIntent(phase < 3.5 ? 'rest' : phase < 6.5 ? 'right' : phase < 10 ? 'rest' : 'left');
    }

    // --- loop
    var running = false, visible = true, last = 0, carry = 0, frame = 0;
    function tick(now) {
      if (!running) return;
      var dt = Math.min(0.1, (now - last) / 1000); last = now;
      autopilot(now);
      carry += dt * RATE;
      while (carry >= 1) { step(); carry -= 1; }
      decode();
      draw();
      frame = requestAnimationFrame(tick);
    }
    function start() {
      if (running || reducedMotion.matches || !visible || document.hidden) return;
      running = true; last = performance.now(); frame = requestAnimationFrame(tick);
    }
    function stop() { running = false; cancelAnimationFrame(frame); }

    // First frame, drawn synchronously from a pre-filled buffer.
    for (var i = 0; i < N; i++) step();
    readColors(); resize(); draw();
    valueEl.textContent = 'REST'; barEl.style.width = '50%';

    if (window.matchMedia('(hover: none)').matches) {
      var word = document.querySelector('.hover-word');
      if (word) word.textContent = 'Tap';
    }

    themeListeners.push(function () { readColors(); draw(); });
    if ('ResizeObserver' in window) new ResizeObserver(function () { resize(); draw(); }).observe(panel);
    else window.addEventListener('resize', function () { resize(); draw(); });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible) start(); else stop();
      }).observe(panel);
    }
    document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); else start(); });
    if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', function () { if (reducedMotion.matches) stop(); else start(); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
    start();
  }
})();
