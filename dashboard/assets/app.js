/* Замена API дашборда claude.ai для статической публикации.
   Данные берутся из assets/data.js (window.DASH_DATA), который собирает build_data.py. */
(function () {
  const DATA = window.DASH_DATA || {};
  const listeners = [];
  window.dash = {
    data(id) {
      if (!(id in DATA)) return { status: "missing", data: [], columns: [], meta: [] };
      const rows = DATA[id];
      return { status: "ok", data: rows, columns: rows.length ? Object.keys(rows[0]) : [], meta: [], refreshing: false };
    },
    onData(fn) { listeners.push(fn); fn(); },
    setLink(name) { try { history.replaceState(null, "", "#" + name); } catch (e) { /* file:// в некоторых браузерах */ } },
  };
  let timer = null, lastW = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    clearTimeout(timer);
    timer = setTimeout(() => listeners.forEach((fn) => fn()), 120);
  });
})();

(function () {
  const ru = d3.formatLocale({ decimal: ",", thousands: " ", grouping: [3], currency: ["", ""], minus: "−" });
  const F = {
    int: ru.format(",d"), f1: ru.format(",.1f"), f2: ru.format(",.2f"), f3: ru.format(",.3f"),
    r: ru.format(".3f"), pct: ru.format(".1%"), pct2: ru.format(".2%"), s: ru.format(",.3~f"),
  };
  const C = { green: "#1ed760", greenDim: "#2f7a4a", pink: "#f573a0", blue: "#509bf5", yellow: "#ffc864",
              fg: "#ffffff", muted: "#b3b3b3", dim: "#5a5a5a", grid: "#2a2a2a", card: "#181818" };
  const GROUP = { "исходный": C.muted, "кодированный": C.blue, "сконструированный": C.yellow, "линеаризованный": C.pink };
  const TYPE_RU = { single: "Сингл", album: "Альбом", compilation: "Сборник" };
  const root = document.querySelector(".sp");
  const tip = root.querySelector(".sp-tip");
  const rows = (id) => dash.data(id).data || [];
  const num = (v) => +v;

  // ---------- helpers ----------
  const meas = document.createElement("canvas").getContext("2d");
  function tw(s, size) { meas.font = (size || 11) + "px sans-serif"; return meas.measureText(String(s)).width; }

  function showTip(ev, title, lines) {
    tip.replaceChildren();
    const t = document.createElement("div"); t.className = "sp-tip-t"; t.textContent = title; tip.append(t);
    for (const [k, v, c] of lines) {
      const r = document.createElement("div"); r.className = "sp-tip-r";
      if (c) { const s = document.createElement("i"); s.className = "sp-sw"; s.style.background = c; r.append(s); }
      const a = document.createElement("span"); a.textContent = k;
      const b = document.createElement("b"); b.textContent = v;
      r.append(a, b); tip.append(r);
    }
    tip.hidden = false;
    const rr = root.getBoundingClientRect();
    let x = ev.clientX - rr.left + 14, y = ev.clientY - rr.top + 14;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > rr.width - 8) x = ev.clientX - rr.left - w - 14;
    if (x < 8) x = 8;
    if (y + h > rr.height - 8) y = ev.clientY - rr.top - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  function ready(ids, el) {
    const st = ids.map((id) => dash.data(id).status);
    el.querySelector(".sp-fail")?.remove();
    if (st.every((s) => s === "ok")) { el.classList.remove("dash-skeleton"); return true; }
    d3.select(el).selectAll("svg").remove();
    if (st.some((s) => s !== "loading")) {
      el.classList.remove("dash-skeleton");
      const f = document.createElement("div"); f.className = "sp-fail";
      const dot = document.createElement("span"); dot.className = "sp-fail-dot";
      const t = document.createElement("span"); t.textContent = "Failed to load";
      f.append(dot, t); el.append(f);
    } else el.classList.add("dash-skeleton");
    return false;
  }

  function svgIn(el, h, minW) {
    d3.select(el).selectAll("svg").remove();
    const w = Math.max(minW || 0, el.clientWidth || 600);
    const svg = d3.select(el).append("svg").attr("width", minW && w > el.clientWidth ? w : "100%")
      .attr("height", h).attr("viewBox", `0 0 ${w} ${h}`);
    return [svg, w];
  }

  function yGrid(svg, y, ticks, x0, x1, fmt) {
    const g = svg.append("g");
    g.selectAll("line").data(ticks).join("line").attr("x1", x0).attr("x2", x1)
      .attr("y1", (t) => y(t)).attr("y2", (t) => y(t)).attr("stroke", C.grid).attr("stroke-width", 1);
    g.selectAll("text").data(ticks).join("text").attr("x", x0 - 8).attr("y", (t) => y(t)).attr("dy", "0.32em")
      .attr("text-anchor", "end").text((t) => fmt(t));
  }

  function rowWhere(id, where) {
    const parts = where.split("&").map((p) => p.split("=").map(decodeURIComponent));
    return rows(id).find((r) => parts.every(([k, v]) => String(r[k]) === v));
  }

  // ---------- inline values ----------
  function fillValues() {
    root.querySelectorAll(".sp-v[data-source]").forEach((el) => {
      const id = el.dataset.source, d = dash.data(id);
      if (d.status !== "ok") { el.textContent = "—"; if (d.status !== "loading") el.classList.remove("dash-skeleton"); return; }
      const r = el.dataset.where ? rowWhere(id, el.dataset.where) : rows(id)[+el.dataset.row || 0];
      el.classList.remove("dash-skeleton");
      if (!r) { el.textContent = "—"; return; }
      const v = num(r[el.dataset.field]);
      el.textContent = (F[el.dataset.fmt] || F.s)(v);
    });
  }

  // ---------- 2.1 popularity histogram ----------
  function drawHist() {
    const el = document.getElementById("c-hist");
    if (!ready(["pop_hist"], el)) return;
    const data = rows("pop_hist"), h = 240, m = { t: 10, r: 6, b: 30 };
    const [svg, w] = svgIn(el, h);
    const y = d3.scaleLinear([0, d3.max(data, (d) => num(d.albums))], [h - m.b, m.t]).nice();
    const ticks = y.ticks(4); m.l = d3.max(ticks, (t) => tw(F.int(t))) + 12;
    const x = d3.scaleBand(data.map((d) => d.bin_label), [m.l, w - m.r]).padding(0.16);
    yGrid(svg, y, ticks, m.l, w - m.r, F.int);
    svg.append("g").selectAll("rect").data(data).join("rect")
      .attr("x", (d) => x(d.bin_label)).attr("width", x.bandwidth())
      .attr("y", (d) => y(num(d.albums))).attr("height", (d) => y(0) - y(num(d.albums))).attr("rx", 3)
      .attr("fill", (d, i) => (i === 0 ? C.green : C.greenDim))
      .on("mousemove", (ev, d) => showTip(ev, "Популярность " + d.bin_label, [["Альбомов", F.int(num(d.albums))]]))
      .on("mouseleave", hideTip);
    const every = x.bandwidth() < 22 ? 4 : 2;
    svg.append("g").selectAll("text").data(data.filter((d, i) => i % every === 0)).join("text")
      .attr("x", (d) => x(d.bin_label)).attr("y", h - m.b + 16).text((d) => num(d.bin_start));
  }

  // ---------- 2.1 by type ----------
  function drawType() {
    const el = document.getElementById("c-type");
    if (!ready(["by_type"], el)) return;
    const data = rows("by_type").slice().sort((a, b) => num(b.mean_popularity) - num(a.mean_popularity));
    const h = 240, m = { t: 6, r: 12, b: 26 };
    const [svg, w] = svgIn(el, h);
    m.l = d3.max(data, (d) => tw(TYPE_RU[d.album_type] || d.album_type, 13)) + 14;
    const x = d3.scaleLinear([0, d3.max(data, (d) => num(d.mean_popularity))], [m.l, w - m.r - 4]).nice();
    const y = d3.scaleBand(data.map((d) => d.album_type), [m.t, h - m.b]).padding(0.38);
    yGridV(svg, x, x.ticks(4), m.t, h - m.b, F.int, h - m.b + 16);
    const g = svg.append("g").selectAll("g").data(data).join("g");
    g.append("rect").attr("x", m.l).attr("y", (d) => y(d.album_type)).attr("height", y.bandwidth())
      .attr("width", (d) => x(num(d.mean_popularity)) - m.l).attr("rx", 4)
      .attr("fill", (d) => (d.album_type === "single" ? C.pink : C.green))
      .on("mousemove", (ev, d) => showTip(ev, TYPE_RU[d.album_type], [["Средняя", F.f1(num(d.mean_popularity))], ["Медиана", F.int(num(d.median_popularity))], ["Альбомов", F.int(num(d.albums))]]))
      .on("mouseleave", hideTip);
    g.append("text").attr("x", m.l - 10).attr("y", (d) => y(d.album_type) + y.bandwidth() / 2).attr("dy", "0.32em")
      .attr("text-anchor", "end").style("fill", C.fg).style("font-size", "13px").text((d) => TYPE_RU[d.album_type] || d.album_type);
    g.append("text").attr("x", (d) => x(num(d.mean_popularity)) - 8).attr("y", (d) => y(d.album_type) + y.bandwidth() / 2).attr("dy", "0.32em")
      .attr("text-anchor", "end").style("fill", "#000").style("font-weight", 700).text((d) => F.f1(num(d.mean_popularity)));
    g.append("text").attr("x", m.l).attr("y", (d) => y(d.album_type) - 5)
      .text((d) => `медиана ${F.int(num(d.median_popularity))} · ${F.int(num(d.albums))} альбомов`);
  }

  function yGridV(svg, x, ticks, y0, y1, fmt, labelY) {
    const g = svg.append("g");
    g.selectAll("line").data(ticks).join("line").attr("y1", y0).attr("y2", y1)
      .attr("x1", (t) => x(t)).attr("x2", (t) => x(t)).attr("stroke", C.grid);
    g.selectAll("text").data(ticks).join("text").attr("y", labelY).attr("x", (t) => x(t)).attr("text-anchor", "middle").text(fmt);
  }

  // ---------- 2.1 by year ----------
  function drawYear() {
    const el = document.getElementById("c-year");
    if (!ready(["by_year"], el)) return;
    const data = rows("by_year").map((d) => ({ year: num(d.year), albums: num(d.albums), pop: num(d.mean_popularity) }));
    const h = 320, m = { t: 14, r: 10, b: 24 }, split = 0.58;
    const [svg, w] = svgIn(el, h);
    const yTopRange = [m.t + (h - m.b - m.t) * split - 16, m.t];
    const yP = d3.scaleLinear([0, d3.max(data, (d) => d.pop)], yTopRange).nice();
    const yA = d3.scaleLinear([0, d3.max(data, (d) => d.albums)], [h - m.b, m.t + (h - m.b - m.t) * split + 10]).nice();
    m.l = Math.max(d3.max(yP.ticks(3), (t) => tw(F.int(t))), d3.max(yA.ticks(2), (t) => tw(F.int(t)))) + 12;
    const ext = d3.extent(data, (d) => d.year);
    const x = d3.scaleLinear([ext[0] - 0.5, ext[1] + 0.5], [m.l, w - m.r]);
    const bw = Math.max(1, (x(1) - x(0)) * 0.72);
    yGrid(svg, yP, yP.ticks(3), m.l, w - m.r, F.int);
    yGrid(svg, yA, yA.ticks(2), m.l, w - m.r, F.int);
    svg.append("text").attr("x", m.l).attr("y", m.t - 2).style("fill", C.fg).style("font-weight", 600).text("Средняя популярность");
    svg.append("text").attr("x", m.l).attr("y", yA.range()[1] - 4).style("fill", C.fg).style("font-weight", 600).text("Число альбомов");
    // reference: year with the lowest mean popularity among years with at least 300 albums
    const big = data.filter((d) => d.albums >= 300);
    const ref = big.length ? d3.least(big, (d) => d.pop) : null;
    if (ref) {
      svg.append("line").attr("x1", x(ref.year)).attr("x2", x(ref.year)).attr("y1", m.t).attr("y2", h - m.b)
        .attr("stroke", C.muted).attr("stroke-dasharray", "4 4").attr("opacity", 0.6);
      svg.append("text").attr("x", x(ref.year) + 6).attr("y", yP(ref.pop) + 16).style("fill", C.fg)
        .text(`${ref.year}: ${F.f1(ref.pop)}`);
    }
    svg.append("g").selectAll("rect").data(data).join("rect").attr("x", (d) => x(d.year) - bw / 2).attr("width", bw)
      .attr("y", (d) => yA(d.albums)).attr("height", (d) => yA(0) - yA(d.albums)).attr("rx", Math.min(2, bw / 2)).attr("fill", C.greenDim);
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", C.green).attr("stroke-width", 2)
      .attr("stroke-linejoin", "round").attr("stroke-linecap", "round")
      .attr("d", d3.line().x((d) => x(d.year)).y((d) => yP(d.pop)));
    const ticks = x.ticks(Math.max(3, Math.floor((w - m.l) / 70))).filter((t) => Number.isInteger(t));
    svg.append("g").selectAll("text").data(ticks).join("text").attr("x", (t) => x(t)).attr("y", h - m.b + 16)
      .attr("text-anchor", "middle").text((t) => t);
    const dot = svg.append("circle").attr("r", 4).attr("fill", C.green).attr("stroke", "#000").attr("stroke-width", 1.5).style("display", "none");
    const rule = svg.append("line").attr("y1", m.t).attr("y2", h - m.b).attr("stroke", C.fg).attr("opacity", 0.25).style("display", "none");
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", w - m.r - m.l).attr("height", h - m.b - m.t).attr("fill", "transparent")
      .on("mousemove", (ev) => {
        const [px] = d3.pointer(ev);
        const yr = Math.round(x.invert(px));
        const d = data.find((r) => r.year === yr); if (!d) return;
        dot.style("display", null).attr("cx", x(d.year)).attr("cy", yP(d.pop));
        rule.style("display", null).attr("x1", x(d.year)).attr("x2", x(d.year));
        showTip(ev, String(d.year), [["Средняя популярность", F.f1(d.pop), C.green], ["Альбомов", F.int(d.albums), C.greenDim]]);
      })
      .on("mouseleave", () => { dot.style("display", "none"); rule.style("display", "none"); hideTip(); });
  }

  // ---------- 2.1 top labels ----------
  function drawLabels() {
    const el = document.getElementById("c-labels");
    if (!ready(["top_labels"], el)) return;
    const data = rows("top_labels");
    const rowH = 24, m = { t: 4, r: 64, b: 6 }, h = m.t + m.b + rowH * data.length;
    const [svg, w] = svgIn(el, h);
    const maxLab = Math.min(w * 0.42, d3.max(data, (d) => tw(d.label, 12)) + 12);
    m.l = maxLab;
    const fit = (s) => { let t = String(s); while (t.length > 3 && tw(t, 12) > maxLab - 12) t = t.slice(0, -2); return t === String(s) ? t : t + "…"; };
    const x = d3.scaleLinear([0, d3.max(data, (d) => num(d.albums))], [m.l, w - m.r]);
    const y = d3.scaleBand(data.map((d) => d.label), [m.t, h - m.b]).padding(0.28);
    const g = svg.append("g").selectAll("g").data(data).join("g")
      .on("mousemove", (ev, d) => showTip(ev, d.label, [["Альбомов", F.int(num(d.albums))], ["Средняя популярность", F.f1(num(d.mean_popularity))], ["Тип", +d.is_karaoke ? "караоке" : "обычный"]]))
      .on("mouseleave", hideTip);
    g.append("text").attr("x", m.l - 10).attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.32em").attr("text-anchor", "end")
      .style("fill", C.fg).style("font-size", "12px").text((d) => fit(d.label));
    g.append("rect").attr("x", m.l).attr("y", (d) => y(d.label)).attr("height", y.bandwidth())
      .attr("width", (d) => x(num(d.albums)) - m.l).attr("rx", 4).attr("fill", (d) => (+d.is_karaoke ? C.pink : C.green));
    g.append("text").attr("x", (d) => x(num(d.albums)) + 6).attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.32em")
      .text((d) => `${F.int(num(d.albums))} · ${F.f1(num(d.mean_popularity))}`);
  }

  // ---------- 2.3 flags (dumbbell) ----------
  function drawFlags() {
    const el = document.getElementById("c-flags");
    if (!ready(["flags"], el)) return;
    const data = rows("flags");
    const rowH = 54, m = { t: 6, r: 22, b: 26 }, h = m.t + m.b + rowH * data.length;
    const [svg, w] = svgIn(el, h);
    m.l = 16;
    const xmax = d3.max(data, (d) => Math.max(num(d.popularity_with), num(d.popularity_without)));
    const x = d3.scaleLinear([0, xmax], [m.l, w - m.r]).nice();
    const y = d3.scaleBand(data.map((d) => d.feature), [m.t, h - m.b]).padding(0);
    yGridV(svg, x, x.ticks(5), m.t, h - m.b, F.int, h - m.b + 16);
    const g = svg.append("g").selectAll("g").data(data).join("g").attr("transform", (d) => `translate(0,${y(d.feature)})`)
      .on("mousemove", (ev, d) => showTip(ev, d.name, [["С признаком", F.f1(num(d.popularity_with))], ["Без признака", F.f1(num(d.popularity_without))], ["Доля альбомов", F.pct(num(d.share))]]))
      .on("mouseleave", hideTip);
    g.append("rect").attr("x", m.l).attr("width", w - m.l - m.r).attr("height", rowH).attr("fill", "transparent");
    g.append("text").attr("x", m.l).attr("y", 16).style("fill", C.fg).style("font-size", "12px").style("font-weight", 600)
      .text((d) => `${d.name} (${F.pct(num(d.share))})`);
    const cy = 36;
    const col = (d) => (num(d.popularity_with) >= num(d.popularity_without) ? C.green : C.pink);
    g.append("line").attr("x1", (d) => x(num(d.popularity_without))).attr("x2", (d) => x(num(d.popularity_with)))
      .attr("y1", cy).attr("y2", cy).attr("stroke", "#444").attr("stroke-width", 3).attr("stroke-linecap", "round");
    g.append("circle").attr("cx", (d) => x(num(d.popularity_without))).attr("cy", cy).attr("r", 6).attr("fill", C.dim);
    g.append("circle").attr("cx", (d) => x(num(d.popularity_with))).attr("cy", cy).attr("r", 7).attr("fill", col);
    g.append("text").attr("x", (d) => x(num(d.popularity_with))).attr("y", cy).attr("dy", "0.32em")
      .attr("dx", (d) => (num(d.popularity_with) >= num(d.popularity_without) ? 12 : -12))
      .attr("text-anchor", (d) => (num(d.popularity_with) >= num(d.popularity_without) ? "start" : "end"))
      .style("fill", col).style("font-weight", 700).text((d) => F.f1(num(d.popularity_with)));
  }

  // ---------- 2.3 feat bins ----------
  function drawFeat() {
    const el = document.getElementById("c-feat");
    if (!ready(["feat_bins"], el)) return;
    const data = rows("feat_bins"), h = 240, m = { t: 18, r: 6, b: 30 };
    const [svg, w] = svgIn(el, h);
    const y = d3.scaleLinear([0, d3.max(data, (d) => num(d.mean_popularity))], [h - m.b, m.t]).nice();
    const ticks = y.ticks(4); m.l = d3.max(ticks, (t) => tw(F.int(t))) + 12;
    const x = d3.scaleBand(data.map((d) => d.feat_share_bin), [m.l, w - m.r]).padding(0.28);
    yGrid(svg, y, ticks, m.l, w - m.r, F.int);
    const g = svg.append("g").selectAll("g").data(data).join("g")
      .on("mousemove", (ev, d) => showTip(ev, "Доля треков с feat.: " + d.feat_share_bin, [["Средняя популярность", F.f1(num(d.mean_popularity))], ["Альбомов", F.int(num(d.albums))]]))
      .on("mouseleave", hideTip);
    g.append("rect").attr("x", (d) => x(d.feat_share_bin)).attr("width", x.bandwidth())
      .attr("y", (d) => y(num(d.mean_popularity))).attr("height", (d) => y(0) - y(num(d.mean_popularity))).attr("rx", 4)
      .attr("fill", (d) => (d.feat_share_bin === "0" ? C.dim : C.green));
    g.append("text").attr("x", (d) => x(d.feat_share_bin) + x.bandwidth() / 2).attr("y", (d) => y(num(d.mean_popularity)) - 6)
      .attr("text-anchor", "middle").style("fill", C.fg).text((d) => F.f1(num(d.mean_popularity)));
    g.append("text").attr("x", (d) => x(d.feat_share_bin) + x.bandwidth() / 2).attr("y", h - m.b + 16)
      .attr("text-anchor", "middle").text((d) => d.feat_share_bin);
  }

  // ---------- tables ----------
  function fillTable(tableId, id, cols) {
    const table = document.getElementById(tableId), tb = table.querySelector("tbody");
    tb.replaceChildren();
    const d = dash.data(id);
    if (d.status !== "ok") {
      const tr = document.createElement("tr"), td = document.createElement("td");
      td.colSpan = cols.length; td.className = "mut";
      td.textContent = d.status === "loading" ? "Загрузка…" : "Failed to load";
      tr.append(td); tb.append(tr); return;
    }
    for (const r of rows(id)) {
      const tr = document.createElement("tr");
      if (table.dataset.rowKey === "key") tr.dataset.key = r.key;
      for (const c of cols) {
        const td = document.createElement("td");
        if (c.cls) td.className = c.cls;
        const v = r[c.f];
        if (c.render) c.render(td, v, r); else td.textContent = c.fmt ? c.fmt(num(v)) : String(v);
        tr.append(td);
      }
      tb.append(tr);
    }
  }
  const signed = (td, v) => { const n = num(v); td.textContent = F.r(n); td.classList.add(n >= 0 ? "sp-pos" : "sp-neg"); };

  function tables() {
    fillTable("t-enc", "encoding", [
      { f: "feature", cls: "code" }, { f: "kind", cls: "mut" }, { f: "categories", cls: "n", fmt: F.int },
      { f: "method" }, { f: "reason", cls: "mut" }]);
    fillTable("t-hyp", "hyp", [
      { f: "feature", cls: "code" }, { f: "hypothesis" }, { f: "expected", cls: "n" },
      { f: "pearson", cls: "n", render: signed }, { f: "spearman", cls: "n", render: signed },
      { f: "verdict", render: (td, v) => { const s = document.createElement("span"); s.className = "sp-tag " + (v === "подтверждена" ? "ok" : v === "частично" ? "mid" : "bad"); s.textContent = v; td.append(s); } }]);
    fillTable("t-lin", "lin_compare", [
      { f: "feature", cls: "code" }, { f: "transform", cls: "code" },
      { f: "pearson_before", cls: "n", render: signed }, { f: "pearson_after", cls: "n", render: signed },
      { f: "spearman_before", cls: "n", render: signed }, { f: "spearman_after", cls: "n", render: signed },
      { f: "trend_r2_before", cls: "n", fmt: F.f2 }, { f: "trend_r2_after", cls: "n", fmt: F.f2 }]);
    fillTable("t-scale", "scale", [
      { f: "feature", cls: "code" }, { f: "method" },
      ...["mean", "std", "min", "median", "max", "iqr"].map((f) => ({ f, cls: "n", fmt: F.s })),
      { f: "skew", cls: "n", fmt: F.f2 }, { f: "test_mean", cls: "n", fmt: F.s }, { f: "test_std", cls: "n", fmt: F.s }]);
  }

  // ---------- 2.4 linearization ----------
  function drawLin() {
    const ok = dash.data("lin_trend").status === "ok";
    const all = ok ? rows("lin_trend") : [];
    root.querySelectorAll(".sp-lincard").forEach((card) => {
      const feat = card.dataset.feature;
      card.querySelectorAll(".sp-chart[data-stage]").forEach((el) => {
        if (!ready(["lin_trend"], el)) return;
        const stage = el.dataset.stage, after = stage === "после";
        const data = all.filter((r) => r.feature === feat && r.stage === stage)
          .map((r) => ({ x: num(r.x), y: num(r.mean_popularity), n: num(r.albums) })).sort((a, b) => a.x - b.x);
        const h = 150, m = { t: 8, r: 8, b: 22 };
        const [svg, w] = svgIn(el, h);
        const y = d3.scaleLinear([0, d3.max(all.filter((r) => r.feature === feat), (r) => num(r.mean_popularity))], [h - m.b, m.t]).nice();
        const yt = y.ticks(3); m.l = d3.max(yt, (t) => tw(F.int(t))) + 10;
        const x = d3.scaleLinear(d3.extent(data, (d) => d.x), [m.l, w - m.r]);
        yGrid(svg, y, yt, m.l, w - m.r, F.int);
        const xt = x.ticks(3);
        svg.append("g").selectAll("text").data(xt).join("text").attr("x", (t) => x(t)).attr("y", h - m.b + 15)
          .attr("text-anchor", "middle").text((t) => (feat === "release_year" && !after ? String(t) : F.s(t)));
        const color = after ? C.green : C.fg;
        svg.append("path").datum(data).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2)
          .attr("stroke-opacity", after ? 1 : 0.7).attr("stroke-linejoin", "round").attr("stroke-linecap", "round")
          .attr("d", d3.line().x((d) => x(d.x)).y((d) => y(d.y)));
        if (data.length > 1) {
          const xs = data.map((d) => d.x), ys = data.map((d) => d.y), ws = data.map((d) => d.n);
          const sw = d3.sum(ws), mx = d3.sum(xs.map((v, i) => v * ws[i])) / sw, my = d3.sum(ys.map((v, i) => v * ws[i])) / sw;
          const k = d3.sum(xs.map((v, i) => ws[i] * (v - mx) * (ys[i] - my))) / d3.sum(xs.map((v, i) => ws[i] * (v - mx) ** 2));
          const x0 = x.domain()[0], x1 = x.domain()[1];
          svg.append("line").attr("x1", x(x0)).attr("x2", x(x1)).attr("y1", y(my + k * (x0 - mx))).attr("y2", y(my + k * (x1 - mx)))
            .attr("stroke", C.muted).attr("stroke-dasharray", "4 4").attr("opacity", 0.55);
        }
        svg.append("g").selectAll("circle").data(data).join("circle").attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
          .attr("r", 2.6).attr("fill", color)
          .on("mousemove", (ev, d) => showTip(ev, `${after ? "После" : "До"} · x = ${F.s(d.x)}`, [["Средняя популярность", F.f1(d.y)], ["Альбомов в интервале", F.int(d.n)]]))
          .on("mouseleave", hideTip);
      });
    });
  }

  // ---------- 2.6 target correlations ----------
  function drawTarget() {
    const el = document.getElementById("c-target");
    if (!ready(["target_corr"], el)) return;
    const data = rows("target_corr").map((d) => ({ f: d.feature, g: d.group, p: num(d.pearson), s: num(d.spearman) }))
      .sort((a, b) => Math.abs(b.s) - Math.abs(a.s));
    const rowH = 23, m = { t: 8, r: 16, b: 26 }, h = m.t + m.b + rowH * data.length;
    const [svg, w] = svgIn(el, h);
    m.l = d3.max(data, (d) => tw(d.f, 12)) + 16;
    const lo = Math.min(0, d3.min(data, (d) => Math.min(d.p, d.s))), hi = Math.max(0, d3.max(data, (d) => Math.max(d.p, d.s)));
    const x = d3.scaleLinear([lo, hi], [m.l, w - m.r]).nice();
    const y = d3.scaleBand(data.map((d) => d.f), [m.t, h - m.b]).padding(0.2);
    yGridV(svg, x, x.ticks(6), m.t, h - m.b, F.f1, h - m.b + 16);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", C.muted).attr("opacity", 0.6);
    const g = svg.append("g").selectAll("g").data(data).join("g").attr("transform", (d) => `translate(0,${y(d.f) + y.bandwidth() / 2})`)
      .on("mousemove", (ev, d) => showTip(ev, `${d.f} · ${d.g}`, [["Пирсон", F.r(d.p), C.fg], ["Спирмен", F.r(d.s), C.green], ["Разность", F.r(d.s - d.p)]]))
      .on("mouseleave", hideTip);
    g.append("rect").attr("x", 0).attr("y", -y.bandwidth() / 2).attr("width", w).attr("height", y.bandwidth()).attr("fill", "transparent");
    g.append("text").attr("x", m.l - 10).attr("dy", "0.32em").attr("text-anchor", "end").style("font-size", "12px")
      .style("fill", (d) => GROUP[d.g] || C.muted).text((d) => d.f);
    g.append("line").attr("x1", (d) => x(d.p)).attr("x2", (d) => x(d.s)).attr("stroke", "#4a4a4a").attr("stroke-width", 4).attr("stroke-linecap", "round");
    g.append("circle").attr("cx", (d) => x(d.p)).attr("r", 5).attr("fill", C.card).attr("stroke", C.fg).attr("stroke-width", 2);
    g.append("circle").attr("cx", (d) => x(d.s)).attr("r", 5.5).attr("fill", C.green);
  }

  // ---------- 2.6 heatmap ----------
  let heatMode = "spearman";
  function drawHeat() {
    const el = document.getElementById("c-heat");
    if (!ready(["corr"], el)) return;
    const data = rows("corr").map((d) => ({ a: d.feature_a, b: d.feature_b, i: num(d.order_a), j: num(d.order_b), p: num(d.pearson), s: num(d.spearman) }));
    const names = [];
    data.forEach((d) => { names[d.i] = d.a; });
    const n = names.length;
    const labW = d3.max(names, (s) => tw(s, 11)) + 10;
    const avail = (el.parentElement.clientWidth || 700) - labW - 8;
    const cell = Math.max(18, Math.min(36, Math.floor(avail / n)));
    const top = labW * 0.75 + 6, W = labW + cell * n + labW * 0.65 + 8, H = top + cell * n + 6;
    d3.select(el).selectAll("svg").remove();
    const svg = d3.select(el).append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const val = (d) => (heatMode === "pearson" ? d.p : heatMode === "spearman" ? d.s : d.s - d.p);
    const lim = heatMode === "diff" ? 0.7 : 1;
    const neg = heatMode === "diff" ? C.blue : C.pink, pos = heatMode === "diff" ? C.yellow : C.green;
    const color = (v) => (v < 0 ? d3.interpolateRgb("#202020", neg)(Math.min(1, -v / lim)) : d3.interpolateRgb("#202020", pos)(Math.min(1, v / lim)));
    svg.append("g").selectAll("text").data(names).join("text").attr("x", labW - 6).attr("y", (s, i) => top + i * cell + cell / 2)
      .attr("dy", "0.32em").attr("text-anchor", "end").style("font-size", "11px").text((s) => s);
    svg.append("g").selectAll("text").data(names).join("text")
      .attr("transform", (s, j) => `translate(${labW + j * cell + cell / 2},${top - 6}) rotate(-50)`)
      .style("font-size", "11px").text((s) => s);
    const cells = svg.append("g").selectAll("g").data(data).join("g").attr("transform", (d) => `translate(${labW + d.j * cell},${top + d.i * cell})`)
      .on("mousemove", (ev, d) => showTip(ev, `${d.a} × ${d.b}`, [["Пирсон", F.r(d.p)], ["Спирмен", F.r(d.s)], ["Спирмен − Пирсон", F.r(d.s - d.p)]]))
      .on("mouseleave", hideTip);
    cells.append("rect").attr("width", cell - 1).attr("height", cell - 1).attr("rx", 2).attr("fill", (d) => (d.i === d.j && heatMode !== "diff" ? "#3a3a3a" : color(val(d))));
    if (cell >= 30) {
      cells.filter((d) => d.i !== d.j && Math.abs(val(d)) >= 0.1).append("text").attr("x", (cell - 1) / 2).attr("y", (cell - 1) / 2).attr("dy", "0.32em")
        .attr("text-anchor", "middle").style("font-size", "9px")
        .style("fill", (d) => (Math.abs(val(d)) / lim > 0.55 ? "#000" : C.fg))
        .text((d) => ru.format(".2f")(val(d)));
    }
  }
  root.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    heatMode = b.dataset.mode;
    root.querySelectorAll("[data-mode]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    drawHeat();
  }));

  // ---------- 2.7 scaling range charts ----------
  function drawScale() {
    const ok = dash.data("scale").status === "ok", all = ok ? rows("scale") : [];
    const methods = ["StandardScaler", "MinMaxScaler", "RobustScaler"];
    const mc = { StandardScaler: C.green, MinMaxScaler: C.yellow, RobustScaler: C.blue };
    root.querySelectorAll(".sp-range .sp-card[data-feature]").forEach((card) => {
      const el = card.querySelector(".sp-chart");
      if (!ready(["scale"], el)) return;
      const feat = card.dataset.feature;
      const data = methods.map((mth) => all.find((r) => r.feature === feat && r.method === mth)).filter(Boolean)
        .map((r) => ({ m: r.method, min: num(r.min), max: num(r.max), med: num(r.median), mean: num(r.mean) }));
      const rowH = 34, mg = { t: 6, r: 14, b: 24 }, h = mg.t + mg.b + rowH * data.length;
      const [svg, w] = svgIn(el, h);
      mg.l = d3.max(data, (d) => tw(d.m.replace("Scaler", ""), 12)) + 14;
      const x = d3.scaleLinear([d3.min(data, (d) => d.min), d3.max(data, (d) => d.max)], [mg.l, w - mg.r]).nice();
      const y = d3.scaleBand(data.map((d) => d.m), [mg.t, h - mg.b]).padding(0.3);
      yGridV(svg, x, x.ticks(5), mg.t, h - mg.b, F.s, h - mg.b + 16);
      if (x.domain()[0] < 0) svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", mg.t).attr("y2", h - mg.b).attr("stroke", C.muted).attr("opacity", 0.5);
      const g = svg.append("g").selectAll("g").data(data).join("g").attr("transform", (d) => `translate(0,${y(d.m) + y.bandwidth() / 2})`)
        .on("mousemove", (ev, d) => showTip(ev, `${feat} · ${d.m}`, [["Мин.", F.s(d.min)], ["Медиана", F.s(d.med)], ["Среднее", F.s(d.mean)], ["Макс.", F.s(d.max)]]))
        .on("mouseleave", hideTip);
      g.append("rect").attr("x", 0).attr("y", -y.bandwidth() / 2).attr("width", w).attr("height", y.bandwidth()).attr("fill", "transparent");
      g.append("text").attr("x", mg.l - 10).attr("dy", "0.32em").attr("text-anchor", "end").style("fill", C.fg).style("font-size", "12px").text((d) => d.m.replace("Scaler", ""));
      g.append("line").attr("x1", (d) => x(d.min)).attr("x2", (d) => x(d.max)).attr("stroke", (d) => mc[d.m]).attr("stroke-width", 3).attr("stroke-linecap", "round").attr("opacity", 0.55);
      g.append("circle").attr("cx", (d) => x(d.med)).attr("r", 6).attr("fill", (d) => mc[d.m]).attr("stroke", "#000").attr("stroke-width", 1.5);
    });
  }

  // ---------- nav ----------
  root.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => {
    const el = document.getElementById(b.dataset.go);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); dash.setLink(el.id); }
  }));

  let linked = false;
  dash.onData(() => {
    fillValues(); tables();
    drawHist(); drawType(); drawYear(); drawLabels();
    drawFlags(); drawFeat(); drawLin(); drawTarget(); drawHeat(); drawScale();
    if (!linked) {
      const named = location.hash && document.getElementById(location.hash.slice(1));
      if (named && named.classList.contains("sp-sec")) { named.scrollIntoView(); linked = true; }
    }
  });
})();
