const state = { tasks: [], filtered: [], model: null, page: 1, pageSize: 12, selected: null };
const money = value => value == null ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: value < 1 ? 3 : 0 }).format(value);
const percent = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

function card(label, value, note) {
  return `<div class="card"><div class="card-label">${escapeHtml(label)}</div><div class="card-value">${escapeHtml(value)}</div><div class="card-note">${escapeHtml(note)}</div></div>`;
}

async function load() {
  const [tasks, model] = await Promise.all([fetch("/api/tasks").then(r => r.json()), fetch("/api/model").then(r => r.json())]);
  state.tasks = tasks; state.filtered = tasks; state.model = model;
  setupFilters(); renderSummary(); applyFilters(); renderModel();
}

function setupFilters() {
  const addOptions = (id, values) => {
    const select = document.getElementById(id);
    [...new Set(values)].sort().forEach(value => select.insertAdjacentHTML("beforeend", `<option>${escapeHtml(value)}</option>`));
  };
  addOptions("sector-filter", state.tasks.map(t => t.sector));
  addOptions("occupation-filter", state.tasks.map(t => t.occupation));
  ["search", "sector-filter", "occupation-filter", "split-filter"].forEach(id => document.getElementById(id).addEventListener(id === "search" ? "input" : "change", () => { state.page = 1; applyFilters(); }));
  document.getElementById("prev-page").addEventListener("click", () => { state.page--; renderTasks(); });
  document.getElementById("next-page").addEventListener("click", () => { state.page++; renderTasks(); });
  document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => switchTab(button.dataset.tab)));
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(el => el.classList.toggle("active", el.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(el => el.classList.toggle("active", el.id === `tab-${name}`));
  if (name === "model") setTimeout(drawCharts, 0);
}

function renderSummary() {
  const med = values => [...values].sort((a,b) => a-b)[Math.floor(values.length / 2)];
  const recommended = state.tasks.filter(t => t.recommendation_status === "RECOMMEND").length;
  document.getElementById("summary-cards").innerHTML = [
    card("Tasks", state.tasks.length.toLocaleString(), "GDPVal-style prompts"),
    card("Median P50 value", money(med(state.tasks.map(t => t.p50_usd))), "ML forecast of proxy target"),
    card("Median max cost", money(med(state.tasks.map(t => t.max_cost_usd))), "Exactly 10% of P50"),
    card("Recommendation coverage", `${(recommended / state.tasks.length * 100).toFixed(0)}%`, `${recommended} tasks pass constraints`)
  ].join("");
}

function applyFilters() {
  const query = document.getElementById("search").value.toLowerCase().trim();
  const sector = document.getElementById("sector-filter").value;
  const occupation = document.getElementById("occupation-filter").value;
  const split = document.getElementById("split-filter").value;
  state.filtered = state.tasks.filter(t => (!query || `${t.task_id} ${t.sector} ${t.occupation} ${t.prompt_preview}`.toLowerCase().includes(query)) && (!sector || t.sector === sector) && (!occupation || t.occupation === occupation) && (!split || t.split === split));
  renderTasks();
}

function renderTasks() {
  const pages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * state.pageSize;
  const rows = state.filtered.slice(start, start + state.pageSize);
  document.getElementById("result-count").textContent = `${state.filtered.length} results`;
  document.getElementById("page-label").textContent = `Page ${state.page} of ${pages}`;
  document.getElementById("prev-page").disabled = state.page <= 1;
  document.getElementById("next-page").disabled = state.page >= pages;
  document.getElementById("task-rows").innerHTML = rows.map(t => `<tr data-id="${escapeHtml(t.task_id)}" class="${state.selected === t.task_id ? "selected" : ""}"><td><div class="task-main">${escapeHtml(t.occupation)}</div><div class="task-sub">${escapeHtml(t.prompt_preview)}…</div></td><td><span class="pill ${t.split}">${t.split}</span></td><td class="money">${money(t.p50_usd)}</td><td class="money">${money(t.max_cost_usd)}</td><td>${escapeHtml(t.selected_model)}</td></tr>`).join("");
  document.querySelectorAll("#task-rows tr").forEach(row => row.addEventListener("click", () => selectTask(row.dataset.id)));
}

async function selectTask(id) {
  state.selected = id; renderTasks();
  const item = await fetch(`/api/tasks/${encodeURIComponent(id)}`).then(r => r.json());
  const f = item.economic_value_forecast, c = item.cost_policy, a = item.proxy_label.assumptions;
  const selected = item.recommendation.selected;
  const alternatives = item.recommendation.alternatives || [];
  document.getElementById("task-detail").innerHTML = `<div class="detail-content">
    <div class="detail-topline"><div><p class="eyebrow">${escapeHtml(item.sector)}</p><h2>${escapeHtml(item.occupation)}</h2></div><span class="pill ${item.split}">${item.split}</span></div>
    <p class="detail-meta">${escapeHtml(item.task_id)} · ${escapeHtml(item.task_profile.risk_tier)} risk</p>
    <div class="value-band"><div class="value-cell"><span>P10</span><strong>${money(f.p10_usd)}</strong></div><div class="value-cell"><span>P50 forecast</span><strong>${money(f.p50_usd)}</strong></div><div class="value-cell"><span>P90</span><strong>${money(f.p90_usd)}</strong></div></div>
    <div class="warning">${escapeHtml(item.proxy_label.warning)}</div>
    <div class="section-block"><h3>10% cost policy</h3><div class="assumption-grid"><div class="datum"><span>Max cost · 10% of P50</span><strong>${money(c.max_cost_usd)}</strong></div><div class="datum"><span>Operational · 10% of P10</span><strong>${money(c.operational_budget_usd)}</strong></div></div></div>
    <div class="section-block"><h3>Proxy assumptions</h3><div class="assumption-grid"><div class="datum"><span>Human effort</span><strong>${a.estimated_human_hours.toFixed(1)} hours</strong></div><div class="datum"><span>Hourly wage</span><strong>${money(a.hourly_wage_usd)}</strong></div><div class="datum"><span>Employer load</span><strong>${a.employer_load_factor.toFixed(2)}×</strong></div><div class="datum"><span>Realizability</span><strong>${percent(a.realizability_factor)}</strong></div></div></div>
    <div class="section-block"><h3>Model recommendation</h3>${selected ? `<div class="recommend-card"><div class="datum"><span>Selected model</span><strong>${escapeHtml(selected.display_name)}</strong></div><div class="datum"><span>Expected success</span><strong>${percent(selected.expected_success_probability)}</strong></div><div class="datum"><span>Expected cost</span><strong>${money(selected.expected_execution_cost_usd)}</strong></div><div class="datum"><span>P90 cost</span><strong>${money(selected.p90_execution_cost_usd)}</strong></div></div><p class="detail-meta">Heuristic quality score pending real task-model evaluations.</p>` : `<div class="warning">Abstain: ${escapeHtml(item.recommendation.reason)}</div>`}
    ${alternatives.length ? `<p class="detail-meta">Alternatives: ${alternatives.map(x => `${escapeHtml(x.display_name)} (${percent(x.expected_success_probability)})`).join(", ")}</p>` : ""}</div>
    <div class="section-block"><h3>Task prompt</h3><div class="prompt-box">${escapeHtml(item.prompt)}</div></div>
  </div>`;
}

function renderModel() {
  const meta = state.model.metadata, metrics = meta.test_metrics, counts = meta.split_counts;
  document.getElementById("split-cards").innerHTML = [
    card("Train · 70%", counts.train, "Fits encoder and regressor"),
    card("Calibration · 10%", counts.calibration, "Fits interval radius only"),
    card("Test · 20%", counts.test, "Held-out final evaluation"),
    card("Conformal radius", money(meta.conformal_radius_usd), `${meta.conformal_coverage_target_percent}% target interval`)
  ].join("");
  const metricRows = [
    ["MAE", money(metrics.mae_usd)], ["RMSE", money(metrics.rmse_usd)], ["R²", metrics.r2.toFixed(3)],
    ["MAPE", `${metrics.mape_percent.toFixed(1)}%`], ["Interval coverage", `${metrics.interval_coverage_percent.toFixed(1)}%`], ["Features", meta.feature_count]
  ];
  document.getElementById("metric-list").innerHTML = metricRows.map(([a,b]) => `<div class="metric"><span>${escapeHtml(a)}</span><strong>${escapeHtml(b)}</strong></div>`).join("");
  document.getElementById("method-note").textContent = meta.target_warning;
  const source = document.getElementById("catalog-source"); source.textContent = `Source · ${state.model.catalog_snapshot_date}`; source.href = state.model.catalog_source_url;
  document.getElementById("catalog-method").textContent = state.model.catalog_method;
  document.getElementById("catalog-rows").innerHTML = state.model.model_catalog.map(m => `<tr><td><strong>${escapeHtml(m.provider)}</strong></td><td><strong>${escapeHtml(m.display_name)}</strong><div class="task-sub">${escapeHtml(m.model_id)}</div></td><td>${m.capability.toFixed(2)}</td><td>${money(m.input_per_million_usd)}</td><td>${money(m.output_per_million_usd)}</td><td>${m.cached_input_per_million_usd == null ? "—" : money(m.cached_input_per_million_usd)}</td><td>${m.context_tokens.toLocaleString()}</td><td>${m.modalities.join(", ")}</td></tr>`).join("");
}

function canvasSetup(id) {
  const canvas = document.getElementById(id), ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600, height = Number(canvas.getAttribute("height"));
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); return {ctx, width, height};
}

function drawAxes(ctx, width, height, margin, xLabel, yLabel) {
  ctx.strokeStyle = "#bcc5bf"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(margin, 15); ctx.lineTo(margin, height-margin); ctx.lineTo(width-15, height-margin); ctx.stroke();
  ctx.fillStyle = "#68746e"; ctx.font = "11px system-ui"; ctx.fillText(xLabel, width/2-35, height-8); ctx.save(); ctx.translate(11, height/2+35); ctx.rotate(-Math.PI/2); ctx.fillText(yLabel, 0, 0); ctx.restore();
}

function drawScatter() {
  const {ctx,width,height} = canvasSetup("scatter-chart"), points = state.model.test_points, margin = 45;
  drawAxes(ctx,width,height,margin,"Proxy value (USD)","Predicted (USD)");
  const max = Math.max(...points.flatMap(p => [p.actual,p.predicted])) * 1.08;
  const sx = v => margin + v/max*(width-margin-20), sy = v => height-margin-v/max*(height-margin-20);
  ctx.strokeStyle="#d56a31"; ctx.setLineDash([5,5]); ctx.beginPath(); ctx.moveTo(sx(0),sy(0)); ctx.lineTo(sx(max),sy(max)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(23,107,77,.72)"; points.forEach(p => {ctx.beginPath();ctx.arc(sx(p.actual),sy(p.predicted),4,0,Math.PI*2);ctx.fill();});
}

function drawResiduals() {
  const {ctx,width,height}=canvasSetup("residual-chart"), vals=state.model.test_points.map(p=>p.residual), margin=42, bins=10;
  drawAxes(ctx,width,height,margin,"Residual (USD)","Count"); const min=Math.min(...vals), max=Math.max(...vals), span=max-min||1, counts=Array(bins).fill(0);
  vals.forEach(v=>counts[Math.min(bins-1,Math.floor((v-min)/span*bins))]++); const barW=(width-margin-20)/bins, high=Math.max(...counts);
  counts.forEach((count,i)=>{const h=count/high*(height-margin-35);ctx.fillStyle=i<bins/2?"#376d8b":"#d56a31";ctx.fillRect(margin+i*barW+2,height-margin-h,barW-4,h);});
}

function drawImportance() {
  const {ctx,width,height}=canvasSetup("importance-chart"), items=state.model.feature_importance.slice(0,16), left=175, top=18, row=29, max=Math.max(...items.map(i=>i.importance));
  ctx.font="11px system-ui"; items.forEach((item,i)=>{const y=top+i*row;ctx.fillStyle="#526058";ctx.textAlign="right";ctx.fillText(item.feature.length>25?`${item.feature.slice(0,24)}…`:item.feature,left-8,y+13);ctx.fillStyle="#176b4d";ctx.fillRect(left,y,item.importance/max*(width-left-25),17);}); ctx.textAlign="left";
}

function drawCharts() { if (!state.model) return; drawScatter(); drawResiduals(); drawImportance(); }
window.addEventListener("resize", () => { if (document.getElementById("tab-model").classList.contains("active")) drawCharts(); });
load().catch(error => { document.body.innerHTML = `<main><div class="warning"><strong>Application failed to load.</strong><br>${escapeHtml(error.message)}</div></main>`; console.error(error); });
