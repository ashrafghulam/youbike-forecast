// Shared code for the map page (index.html) and the station page (station.html).
(function () {
  const RAW = "https://raw.githubusercontent.com/ashrafghulam/youbike-forecast/main/docs/data/";

  // Four states a rider understands. Order = priority when several apply.
  const STATUS = {
    nobikes:   { label: "No bikes soon",      short: "No bikes",   icon: "❌", color: "#d93b3b",
                 verdict: "Likely <b>no bikes</b> here in one hour. Try a nearby station." },
    noparking: { label: "No parking soon",    short: "No parking", icon: "🅿️", color: "#1c5cab",
                 verdict: "Likely <b>full</b> in one hour, so it may be hard to return a bike here." },
    few:       { label: "Only a few bikes",   short: "Few bikes",  icon: "⚠️", color: "#e8890c",
                 verdict: "Only <b>a few bikes</b> left in one hour. Come early or have a backup." },
    ok:        { label: "Bikes available",    short: "Available",  icon: "✅", color: "#2b9a66",
                 verdict: "<b>Bikes available</b> in one hour. You should be fine." },
  };
  const ORDER = ["nobikes", "noparking", "few", "ok"];

  function status(bikes, usable) {
    const r = Math.round(bikes);
    if (r <= 2) return "nobikes";
    if (usable - r <= 2) return "noparking";
    if (r <= 5) return "few";
    return "ok";
  }

  // A map pin with a bike inside, one per colour.
  function pinUrl(color) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="120" viewBox="0 0 48 60">
      <path d="M24 2C12.4 2 3 11.2 3 22.6 3 38 24 58 24 58S45 38 45 22.6C45 11.2 35.6 2 24 2Z" fill="${color}" stroke="#fff" stroke-width="2.5"/>
      <g fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="15.5" cy="27" r="5.2"/><circle cx="32.5" cy="27" r="5.2"/>
        <path d="M15.5 27l5.5-10h8l3.5 10M21 17l5.5 10M19 13.5h4.5M29 17l-1.5-3.5h3"/>
      </g></svg>`;
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }
  const PINS = Object.fromEntries(ORDER.map(k => [k, pinUrl(STATUS[k].color)]));

  const hexRgb = (hex) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));

  async function getJSON(name) {
    const bust = "?t=" + Math.floor(Date.now() / 60000);
    for (const base of [RAW, "data/"]) {
      try { const r = await fetch(base + name + bust, { cache: "no-store" }); if (r.ok) return await r.json(); } catch (e) {}
    }
    throw new Error("could not load " + name);
  }

  // Load stations + forecast and join them into one list of station objects.
  async function loadData() {
    const [st, fc] = await Promise.all([getJSON("stations.json"), getJSON("forecast.json")]);
    const list = [];
    for (const s of st.stations) {
      const f = fc.s[s.id];
      if (!f) continue;
      s.now = f[0]; s.docks = f[1]; s.fc = f[2]; s.why = f[3] || []; s.hist = f[4] || [];
      s.usable = s.docks == null ? s.cap : s.now + s.docks;
      s.change = s.fc - s.now;
      s.st = status(s.fc, s.usable);
      s.stNow = status(s.now, s.usable);
      // bikes one hour ago = 6 slots before the last history point
      const h = s.hist, i = h.length - 7;
      s.ago = i >= 0 && h[i] != null ? h[i] : null;
      s.key = normalize(`${s.en} ${s.zh} ${s.d}`);
      s.words = new Set(s.key.split(" "));
      list.push(s);
    }
    return { stations: list, byId: Object.fromEntries(list.map(s => [s.id, s])), meta: fc };
  }

  function normalize(t) {
    return String(t).toLowerCase().replace(/[().,·\-_/\[\]#]+/g, " ").replace(/\s+/g, " ").trim();
  }

  // Every typed word must appear; names that start with the query come first.
  function search(stations, query, limit = 8) {
    const q = normalize(query);
    if (!q) return [];
    if (/^\d{6,}$/.test(q)) return stations.filter(s => s.id.startsWith(q)).slice(0, limit);   // station number
    const words = q.split(" ");
    // a number must match a whole number in the name ("exit 5" should not match "exit 3" or "Ln. 500")
    const ok = (s, w) => /^\d+$/.test(w) ? (s.words.has(w) || s.zh.includes(w + "號")) : s.key.includes(w);
    const hits = [];
    for (const s of stations) {
      if (!words.every(w => ok(s, w))) continue;
      const en = normalize(s.en);
      const score = en.startsWith(q) || s.zh.startsWith(query.trim()) ? 0 : en.includes(q) ? 1 : 2;
      hits.push([score, s.en.length, s]);
    }
    hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return hits.slice(0, limit).map(h => h[2]);
  }

  function distanceM(a, b) {
    const R = 6371000, r = Math.PI / 180;
    const dLat = (b.lat - a.lat) * r, dLon = (b.lon - a.lon) * r;
    const x = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(x));
  }
  function nearest(stations, point, n = 5, exceptId = null) {
    return stations.filter(s => s.id !== exceptId)
      .map(s => [distanceM(point, s), s]).sort((a, b) => a[0] - b[0]).slice(0, n)
      .map(([d, s]) => ({ s, d }));
  }
  const fmtDist = (m) => m < 1000 ? `${Math.round(m / 10) * 10} m` : `${(m / 1000).toFixed(1)} km`;

  // Plain-language reasons. Group order matches web/make_forecast.py REASONS.
  const WHY = [
    ["How full it is now points to fewer bikes", "How empty it is now points to bikes coming back"],
    ["Bikes have been leaving in the last hour", "Bikes have been arriving in the last hour"],
    ["Yesterday at this time, bikes were leaving", "Yesterday at this time, bikes were arriving"],
    ["Nearby stations are emptying too", "Nearby stations are filling up"],
    ["This time of day usually means fewer bikes here", "This time of day usually means more bikes here"],
    ["This area tends to lose bikes now", "This area tends to gain bikes now"],
    ["The weather points to fewer bikes", "The weather points to more bikes"],
  ];
  function whySentences(s) {
    const out = s.why.filter(([, v]) => Math.abs(v) >= 0.3)
      .map(([g, v]) => ({ text: WHY[g][v < 0 ? 0 : 1], v }));
    return out.length ? out : [{ text: "No strong signal. The count should stay about the same.", v: 0 }];
  }

  const TZ = "Asia/Taipei";
  const fmtHM = (iso) => new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: TZ });
  const fmtDay = (iso) => new Date(iso).toLocaleString("en-GB", { weekday: "short", hour: "2-digit", minute: "2-digit", timeZone: TZ });
  const esc = (t) => String(t).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Search box with a suggestion list. onPick(station) is called on choice.
  function attachSearch(input, box, getStations, onPick) {
    let items = [], idx = -1;
    const render = () => {
      if (!items.length) { box.hidden = true; box.innerHTML = ""; return; }
      box.innerHTML = items.map((s, i) => `<li role="option" data-i="${i}" class="${i === idx ? "on" : ""}">
        <span class="sg-dot" style="background:${STATUS[s.st].color}"></span>
        <span class="sg-txt"><b>${esc(s.en)}</b><small>${esc(s.zh)} · ${esc(s.d)}</small></span>
        <span class="sg-n">${s.now} → ${Math.round(s.fc)}</span></li>`).join("");
      box.hidden = false;
    };
    const pick = (s) => { box.hidden = true; input.value = s.en; input.blur(); onPick(s); };
    input.addEventListener("input", () => { items = search(getStations(), input.value); idx = items.length ? 0 : -1; render(); });
    input.addEventListener("keydown", (e) => {
      if (box.hidden) return;
      if (e.key === "ArrowDown") { idx = Math.min(idx + 1, items.length - 1); render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { idx = Math.max(idx - 1, 0); render(); e.preventDefault(); }
      else if (e.key === "Enter" && items[idx]) { pick(items[idx]); e.preventDefault(); }
      else if (e.key === "Escape") { box.hidden = true; }
    });
    box.addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pick(items[+li.dataset.i]); } });
    input.addEventListener("blur", () => setTimeout(() => { box.hidden = true; }, 150));
    input.addEventListener("focus", () => { if (items.length) render(); });
  }

  window.YB = { STATUS, ORDER, PINS, hexRgb, status, loadData, search, nearest, distanceM, fmtDist,
                whySentences, fmtHM, fmtDay, esc, attachSearch };
})();
