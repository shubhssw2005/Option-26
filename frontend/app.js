const API = 'http://localhost:8000';
let charts = {};

// ── Navigation ────────────────────────────────────────────────────────────────
function show(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelectorAll('.nav-item').forEach(n => {
        if (n.getAttribute('onclick')?.includes(page)) n.classList.add('active');
    });
    if (page === 'vol') fetchVol();
    if (page === 'greeks') fetchGreeks();
    if (page === 'history') { fetchHistorical(); fetchNormReturns(); }
}

// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(() => {
    document.getElementById('htime').textContent =
        new Date().toLocaleTimeString('en-IN', { hour12: false });
}, 1000);

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
    const badge = document.getElementById('ws-badge');
    const ws = new WebSocket('ws://localhost:8000/ws/live');
    ws.onopen = () => { badge.textContent = 'LIVE'; badge.className = 'hbadge live'; };
    ws.onclose = () => { badge.textContent = 'Disconnected'; badge.className = 'hbadge'; setTimeout(connectWS, 3000); };
    ws.onmessage = e => {
        const d = JSON.parse(e.data);
        renderIndexes(d.indexes || []);
        document.getElementById('hstatus').textContent = 'Updated ' + new Date().toLocaleTimeString();
    };
}

function renderIndexes(indexes) {
    if (!indexes.length) return;
    document.getElementById('index-list').innerHTML = indexes.map(i => {
        const val = (i.index_value / 100).toLocaleString('en-IN', { minimumFractionDigits: 2 });
        const chg = (i.changepercent || 0).toFixed(2);
        const cls = chg >= 0 ? 'up' : 'down';
        return `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f0f0f0">
      <span style="font-weight:600">${i.indexname}</span>
      <div style="text-align:right">
        <div style="font-size:1rem;font-weight:700">${val}</div>
        <div class="${cls}" style="font-size:.72rem">${chg >= 0 ? '+' : ''}${chg}%</div>
      </div></div>`;
    }).join('');
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  const asset = document.getElementById('dash-asset').value;
  const dte   = document.getElementById('dash-dte').value;

  // Vol regime for all assets
  try {
    const r = await fetch(`${API}/vol-forecast-all`);
    const d = await r.json();
    renderVolRegimes(d);
    renderKPIs(d[asset] || {});
  } catch(e) {}

  // Top strategies
  try {
    const r = await fetch(`${API}/strategy/recommend?asset=${asset}&dte=${dte}`);
    const d = await r.json();
    renderTopStrategies(d.strategies?.slice(0,5) || [], d.regime);
  } catch(e) {}
}

function renderKPIs(vf) {
  const kpis = [
    { label:'GARCH 1d Vol', value: vf.garch_vol_1d ? vf.garch_vol_1d+'%' : '—', sub:'Annualised: '+(vf.garch_vol_ann||'—')+'%' },
    { label:'Realized 20d', value: vf.realized_vol_20d ? vf.realized_vol_20d+'%' : '—', sub:'5d: '+(vf.realized_vol_5d||'—')+'%' },
    { label:'IV Percentile', value: vf.iv_percentile ? vf.iv_percentile+'%' : '—', sub:'Regime: '+(vf.vol_regime||'—') },
    { label:'PCR', value: vf.pcr || '—', sub:'VRP: '+(vf.vol_risk_premium||'—') },
  ];
  document.getElementById('kpi-row').innerHTML = kpis.map(k =>
    `<div class="kpi"><div class="kpi-label">${k.label}</div>
     <div class="kpi-value">${k.value}</div>
     <div class="kpi-sub">${k.sub}</div></div>`
  ).join('');
}

function renderVolRegimes(data) {
  const el = document.getElementById('vol-regime-list');
  el.innerHTML = Object.entries(data).map(([asset, vf]) => {
    if (vf.error) return '';
    const regime = vf.vol_regime || 'fair';
    const ivp = vf.iv_percentile || 50;
    const color = regime==='cheap'?'#1a7a3c':regime==='expensive'?'#c0392b':'#b8860b';
    const pct = Math.min(ivp, 100);
    return `<div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-weight:600;font-size:.8rem">${asset}</span>
        <span style="font-size:.72rem;color:${color};font-weight:700;text-transform:uppercase">${regime} (${ivp}%)</span>
      </div>
      <div class="gauge-bar"><div class="gauge-fill gauge-${regime}" style="width:${pct}%"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#888;margin-top:2px">
        <span>GARCH: ${vf.garch_vol_1d||'—'}%</span>
        <span>RV20: ${vf.realized_vol_20d||'—'}%</span>
        <span>PCR: ${vf.pcr||'—'}</span>
      </div></div>`;
  }).join('');
}

function renderTopStrategies(strategies, regime) {
  const el = document.getElementById('top-strategies');
  if (!strategies.length) { el.innerHTML='<p style="color:#999">No recommendations</p>'; return; }
  el.innerHTML = strategies.map((s,i) => {
    const scoreColor = s.score > 0.6 ? '#1a7a3c' : s.score > 0.4 ? '#b8860b' : '#c0392b';
    const reasons = (s.reasons||[]).map(r => `<div class="strat-reason">${r}</div>`).join('');
    const warnings = (s.warnings||[]).map(w => `<div class="strat-warning">${w}</div>`).join('');
    return `<div class="strat-card ${i===0?'top':''}">
      <div class="strat-score" style="color:${scoreColor}">${(s.score*100).toFixed(0)}</div>
      <div class="strat-name">${s.name}</div>
      <div class="strat-meta">
        <span class="tag tag-${s.direction.includes('bull')?'green':s.direction.includes('bear')?'red':'blue'}">${s.direction}</span>
        <span class="tag tag-${s.vol_view==='long_vol'?'green':s.vol_view==='short_vol'?'red':'grey'}">${s.vol_view}</span>
        <span class="tag tag-grey">${s.category}</span>
        <span class="tag tag-${s.risk_score<=2?'green':s.risk_score>=4?'red':'gold'}">Risk ${s.risk_score}/5</span>
        ${s.margin_required?'<span class="tag tag-gold">Margin</span>':''}
      </div>
      <div class="strat-reasons">${reasons}${warnings}</div>
      <div style="font-size:.72rem;color:#555">
        <strong>Best when:</strong> ${s.best_when}<br>
        <strong>Max loss:</strong> ${s.max_loss} &nbsp;|&nbsp; <strong>Max profit:</strong> ${s.max_profit}
      </div>
      <button class="outline" style="margin-top:8px;font-size:.7rem" onclick="calcStrikes('${s.key}')">Calculate Strikes</button>
    </div>`;
  }).join('');
}

// ── Option Chain ──────────────────────────────────────────────────────────────
async function fetchChain() {
  const sym = document.getElementById('oc-asset').value;
  const exc = document.getElementById('oc-exchange').value;
  const exp = document.getElementById('oc-expiry').value;
  if (!exp) { alert('Enter expiry (YYYYMMDD)'); return; }
  const r = await fetch(`${API}/option-chain?instrument=${sym}&exchange=${exc}&expiry=${exp}`);
  const d = await r.json();
  const chain = d.chain || {};
  const atm = chain.atm ? chain.atm/100 : null;
  const spot = chain.cp ? chain.cp/100 : null;
  document.getElementById('atm-info').textContent =
    atm ? `ATM: ${atm.toLocaleString('en-IN')}  |  Spot: ${spot?.toLocaleString('en-IN')}` : '';
  renderChainTable('ce-table', chain.ce||[], atm, 'ce');
  renderChainTable('pe-table', chain.pe||[], atm, 'pe');
}

function renderChainTable(id, items, atm, side) {
  const sorted = [...items].sort((a,b)=>(a.sp||0)-(b.sp||0));
  document.getElementById(id).innerHTML = sorted.slice(0,30).map(i => {
    const strike = (i.sp/100).toFixed(0);
    const isAtm  = atm && Math.abs(i.sp/100 - atm) < 50;
    const ltp    = i.ltp != null ? (i.ltp/100).toFixed(2) : '—';
    const iv     = i.iv  != null ? (i.iv*100).toFixed(1)  : '—';
    return `<tr class="${isAtm?'atm-row':''}">
      <td><strong>${Number(strike).toLocaleString('en-IN')}</strong></td>
      <td>${ltp}</td><td>${iv}</td>
      <td>${i.delta?.toFixed(3)||'—'}</td>
      <td>${i.gamma?.toFixed(5)||'—'}</td>
      <td>${i.theta?.toFixed(2)||'—'}</td>
      <td>${i.vega?.toFixed(2)||'—'}</td>
      <td>${i.oi?.toLocaleString()||'—'}</td>
      <td>${i.volume?.toLocaleString()||'—'}</td>
    </tr>`;
  }).join('');
}

// ── Volatility ────────────────────────────────────────────────────────────────
let garchChart=null, forecastChart=null;

async function fetchVol() {
  const asset = document.getElementById('vol-asset').value;
  const r = await fetch(`${API}/vol-forecast-v2?asset=${asset}`);
  const d = await r.json();
  if (d.error) return;

  // KPIs
  document.getElementById('vol-kpis').innerHTML = [
    {label:'GARCH 1d',    value:d.garch_vol_1d+'%',    sub:'Annualised: '+d.garch_vol_ann+'%'},
    {label:'Realized 20d',value:d.realized_vol_20d+'%', sub:'5d: '+d.realized_vol_5d+'%'},
    {label:'Current IV',  value:(d.current_iv_pct||'—')+'%', sub:'Percentile: '+d.iv_percentile+'%'},
    {label:'Vol Risk Prem',value:(d.vol_risk_premium||'—')+'%', sub:'PCR: '+d.pcr},
  ].map(k=>`<div class="kpi"><div class="kpi-label">${k.label}</div>
    <div class="kpi-value">${k.value}</div><div class="kpi-sub">${k.sub}</div></div>`).join('');

  // GARCH comparison bar chart
  const models = d.models || {};
  const mNames = Object.keys(models);
  const mVols  = mNames.map(m => models[m].vol_1d || 0);
  const ctx1 = document.getElementById('garch-chart').getContext('2d');
  if (garchChart) garchChart.destroy();
  garchChart = new Chart(ctx1, {
    type:'bar',
    data:{labels:mNames, datasets:[{label:'1-Day Vol%', data:mVols,
      backgroundColor:['#111','#333','#555'], borderRadius:2}]},
    options:{animation:false, plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#555'}}, y:{ticks:{color:'#555'}, title:{display:true,text:'Vol %'}}}}
  });

  // 5-day forecast
  const sarima = d.sarima_forecast || [];
  const ctx2 = document.getElementById('forecast-chart').getContext('2d');
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx2, {
    type:'line',
    data:{labels:['D+1','D+2','D+3','D+4','D+5'],
      datasets:[{label:'SARIMA Return Forecast', data:sarima,
        borderColor:'#111', borderWidth:2, pointRadius:4, fill:false}]},
    options:{animation:false, plugins:{legend:{labels:{color:'#111'}}},
      scales:{x:{ticks:{color:'#555'}}, y:{ticks:{color:'#555'}}}}
  });

  // SARIMA
  document.getElementById('sarima-result').innerHTML = sarima.length
    ? `<div style="font-size:.8rem">Model: <strong>${d.sarima_model}</strong></div>
       <div style="display:flex;gap:12px;margin-top:8px">${sarima.map((v,i)=>
         `<div class="kpi" style="flex:1"><div class="kpi-label">Day ${i+1}</div>
          <div class="kpi-value ${v>=0?'up':'down'}">${v>=0?'+':''}${v}%</div></div>`
       ).join('')}</div>`
    : '<p style="color:#999">No SARIMA forecast available</p>';

  // All assets table
  fetchVolAll();
}

async function fetchVolAll() {
  const r = await fetch(`${API}/vol-forecast-all`);
  const d = await r.json();
  document.getElementById('vol-all-table').innerHTML = Object.entries(d).map(([asset, vf]) => {
    if (vf.error) return `<tr><td>${asset}</td><td colspan="8" style="color:#999">${vf.error}</td></tr>`;
    const regime = vf.vol_regime || 'fair';
    const rc = regime==='cheap'?'tag-green':regime==='expensive'?'tag-red':'tag-gold';
    return `<tr>
      <td><strong>${asset}</strong></td>
      <td>${vf.garch_vol_1d||'—'}</td>
      <td>${vf.garch_vol_ann||'—'}</td>
      <td>${vf.realized_vol_20d||'—'}</td>
      <td>${vf.current_iv_pct||'—'}</td>
      <td>${vf.iv_percentile||'—'}</td>
      <td class="${(vf.vol_risk_premium||0)>0?'up':'down'}">${vf.vol_risk_premium||'—'}</td>
      <td>${vf.pcr||'—'}</td>
      <td><span class="tag ${rc}">${regime}</span></td>
    </tr>`;
  }).join('');
}

// ── Strategy Engine ───────────────────────────────────────────────────────────
async function fetchStrategy() {
  const asset = document.getElementById('st-asset').value;
  const dte   = document.getElementById('st-dte').value;
  const r = await fetch(`${API}/strategy/recommend?asset=${asset}&dte=${dte}`);
  const d = await r.json();

  // Regime display
  const reg = d.regime || {};
  const regClass = `regime-${(reg.regime||'uncertain').replace('_','-')}`;
  document.getElementById('regime-display').innerHTML = `
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
      <span class="regime-badge ${regClass}">${reg.regime||'unknown'}</span>
      <span style="font-size:.75rem">Vol: <strong>${reg.vol_regime||'—'}</strong></span>
      <span style="font-size:.75rem">IV Pct: <strong>${reg.iv_percentile||'—'}%</strong></span>
      <span style="font-size:.75rem">PCR: <strong>${reg.pcr||'—'}</strong></span>
      <span style="font-size:.75rem">GARCH 1d: <strong>${reg.garch_vol_1d||'—'}%</strong></span>
      <span style="font-size:.75rem">CE Signal: <strong>${(d.signal_ce*100||0).toFixed(0)}</strong></span>
      <span style="font-size:.75rem">PE Signal: <strong>${(d.signal_pe*100||0).toFixed(0)}</strong></span>
      <span style="font-size:.75rem">Liquidity: <strong>${(d.liquidity*100||0).toFixed(0)}%</strong></span>
    </div>`;

  // Strategy list
  const strats = d.strategies || [];
  document.getElementById('strategy-list').innerHTML = strats.map((s,i) => {
    const scoreColor = s.score>0.6?'#1a7a3c':s.score>0.4?'#b8860b':'#c0392b';
    const reasons  = (s.reasons||[]).map(r=>`<div class="strat-reason">${r}</div>`).join('');
    const warnings = (s.warnings||[]).map(w=>`<div class="strat-warning">${w}</div>`).join('');
    const greeks = s.net_greeks || {};
    return `<div class="strat-card ${i===0?'top':''}">
      <div class="strat-score" style="color:${scoreColor}">${(s.score*100).toFixed(0)}</div>
      <div class="strat-name">${s.name}</div>
      <div class="strat-meta">
        <span class="tag tag-${s.direction.includes('bull')?'green':s.direction.includes('bear')?'red':'blue'}">${s.direction}</span>
        <span class="tag tag-${s.vol_view==='long_vol'?'green':s.vol_view==='short_vol'?'red':'grey'}">${s.vol_view}</span>
        <span class="tag tag-grey">${s.category}</span>
        <span class="tag tag-${s.risk_score<=2?'green':s.risk_score>=4?'red':'gold'}">Risk ${s.risk_score}/5</span>
        ${s.margin_required?'<span class="tag tag-gold">Margin</span>':''}
      </div>
      <div class="strat-reasons">${reasons}${warnings}</div>
      <div style="font-size:.72rem;color:#555;margin-bottom:8px">
        <strong>Best when:</strong> ${s.best_when}<br>
        <strong>Max loss:</strong> ${s.max_loss} &nbsp;|&nbsp; <strong>Max profit:</strong> ${s.max_profit}<br>
        <strong>Breakeven:</strong> ${s.breakeven}
      </div>
      <div class="greeks-grid" style="margin-bottom:8px">
        ${['delta','gamma','theta','vega'].map(g=>`
          <div class="greek-box">
            <div class="greek-name">${g}</div>
            <div class="greek-val ${(greeks[g]||0)>0?'up':(greeks[g]||0)<0?'down':'neutral'}">${(greeks[g]||0).toFixed(2)}</div>
          </div>`).join('')}
      </div>
      <button class="outline" style="font-size:.7rem" onclick="calcStrikes('${s.key}','${asset}')">Calculate Strikes →</button>
    </div>`;
  }).join('');
}

async function calcStrikes(stratKey, asset) {
  asset = asset || document.getElementById('st-asset').value;
  const r = await fetch(`${API}/strategy/strikes?asset=${asset}&strategy=${stratKey}`);
  const d = await r.json();
  const card = document.getElementById('strike-calc-card');
  card.style.display = 'block';
  if (d.error) { document.getElementById('strike-result').innerHTML=`<p style="color:#c0392b">${d.error}</p>`; return; }

  const legs = (d.legs||[]).map(l => `
    <div class="leg-row">
      <span style="width:50px;font-weight:700;color:${l.side==='long'?'#1a7a3c':'#c0392b'}">${l.side.toUpperCase()}</span>
      <span style="width:40px">${l.type.toUpperCase()}</span>
      <span style="width:80px">K=${l.strike?.toLocaleString('en-IN')}</span>
      <span style="width:70px">₹${l.premium}</span>
      <span style="width:50px">IV ${l.iv_pct}%</span>
      <span style="width:60px">Δ ${l.delta}</span>
      <span style="width:60px">θ ${l.theta}</span>
      <span style="width:60px">ν ${l.vega}</span>
      <span style="width:80px">OI ${l.oi?.toLocaleString()}</span>
      <span style="font-weight:600;color:${l.side==='long'?'#c0392b':'#1a7a3c'}">₹${l.cost?.toLocaleString()}</span>
    </div>`).join('');

  document.getElementById('strike-result').innerHTML = `
    <div style="margin-bottom:12px">
      <strong>${d.strategy}</strong> on ${d.spot?.toLocaleString('en-IN')} (ATM: ${d.atm_strike?.toLocaleString('en-IN')})
    </div>
    <div class="strat-legs">${legs}</div>
    <div style="display:flex;gap:16px;margin-top:12px;font-size:.8rem">
      <div>Net Debit: <strong>₹${d.net_debit?.toLocaleString()}</strong></div>
      <div>Net Credit: <strong>₹${d.net_credit?.toLocaleString()}</strong></div>
      <div>Net Cost: <strong>₹${d.net_cost?.toLocaleString()}</strong></div>
    </div>
    <div style="margin-top:8px;font-size:.75rem;color:#555">
      Max Loss: ${d.max_loss_inr} &nbsp;|&nbsp; Max Profit: ${d.max_profit_inr}<br>
      Breakeven: ${d.breakeven}
    </div>`;
  card.scrollIntoView({behavior:'smooth'});
}

// ── ML Signals ────────────────────────────────────────────────────────────────
async function fetchSignals() {
  const asset = document.getElementById('sig-asset').value;
  const r = await fetch(`${API}/signals?asset=${asset}`);
  const d = await r.json();
  renderSignalTable('ce-signals', d.CE||[], '#1a7a3c');
  renderSignalTable('pe-signals', d.PE||[], '#c0392b');
}

function renderSignalTable(id, rows, color) {
  document.getElementById(id).innerHTML = rows.map(r => {
    const score = r.signal_score || 0;
    const bar = `<div style="display:flex;align-items:center;gap:6px">
      <div style="width:${score*60}px;height:6px;background:${color};border-radius:3px"></div>
      <span style="font-weight:700">${(score*100).toFixed(0)}</span></div>`;
    return `<tr>
      <td>${r.strike?.toLocaleString('en-IN')||'—'}</td>
      <td>${r.ltp?.toFixed(2)||'—'}</td>
      <td>${r.iv ? (r.iv*100).toFixed(1)+'%' : '—'}</td>
      <td>${r.delta?.toFixed(3)||'—'}</td>
      <td>${r.oi?.toLocaleString()||'—'}</td>
      <td>${bar}</td>
    </tr>`;
  }).join('');
}

async function fetchAllSignals() {
  const r = await fetch(`${API}/signals/all`);
  const d = await r.json();
  const card = document.getElementById('all-signals-card');
  card.style.display = 'block';
  document.getElementById('all-signals-content').innerHTML = Object.entries(d).map(([asset, sigs]) => {
    if (sigs.error) return `<div style="margin-bottom:8px"><strong>${asset}</strong>: <span style="color:#999">${sigs.error}</span></div>`;
    const ce = (sigs.CE||[])[0];
    const pe = (sigs.PE||[])[0];
    return `<div style="display:flex;gap:16px;align-items:center;padding:8px 0;border-bottom:1px solid #f0f0f0">
      <span style="font-weight:700;width:100px">${asset}</span>
      <span style="color:#1a7a3c;font-size:.75rem">CE: K=${ce?.strike?.toLocaleString()||'—'} score=${ce?((ce.signal_score||0)*100).toFixed(0):'—'}</span>
      <span style="color:#c0392b;font-size:.75rem">PE: K=${pe?.strike?.toLocaleString()||'—'} score=${pe?((pe.signal_score||0)*100).toFixed(0):'—'}</span>
    </div>`;
  }).join('');
}

// ── Greeks ────────────────────────────────────────────────────────────────────
let greekCharts = {};
async function fetchGreeks() {
  const asset = document.getElementById('gr-asset').value;
  const r = await fetch(`${API}/iv-surface?asset=${asset}`);
  const d = await r.json();
  if (!Array.isArray(d) || !d.length) return;

  const ce = d.filter(x=>x.option_type==='CE').sort((a,b)=>a.strike-b.strike);
  const pe = d.filter(x=>x.option_type==='PE').sort((a,b)=>a.strike-b.strike);
  const labels = [...new Set(d.map(x=>(x.strike/100).toFixed(0)))];

  function mkChart(id, label, ceData, peData) {
    const ctx = document.getElementById(id).getContext('2d');
    if (greekCharts[id]) greekCharts[id].destroy();
    greekCharts[id] = new Chart(ctx, {
      type:'line',
      data:{labels, datasets:[
        {label:'CE '+label, data:ceData, borderColor:'#1a7a3c', borderWidth:1.5, pointRadius:2, fill:false},
        {label:'PE '+label, data:peData, borderColor:'#c0392b', borderWidth:1.5, pointRadius:2, fill:false},
      ]},
      options:{animation:false, plugins:{legend:{labels:{color:'#111',font:{size:10}}}},
        scales:{x:{ticks:{color:'#555',maxTicksLimit:8}}, y:{ticks:{color:'#555'}}}}
    });
  }

  const getField = (arr, field) => arr.map(x => x[field] != null ? x[field] : null);
  mkChart('delta-chart', 'Delta',  getField(ce,'delta'),  getField(pe,'delta'));
  mkChart('gamma-chart', 'Gamma',  getField(ce,'gamma'),  getField(pe,'gamma'));
  mkChart('theta-chart', 'Theta',  getField(ce,'theta'),  getField(pe,'theta'));
  mkChart('vega-chart',  'Vega',   getField(ce,'vega'),   getField(pe,'vega'));

  // IV smile
  const ctx = document.getElementById('iv-smile-chart').getContext('2d');
  if (greekCharts['iv']) greekCharts['iv'].destroy();
  greekCharts['iv'] = new Chart(ctx, {
    type:'line',
    data:{labels, datasets:[
      {label:'CE IV%', data:ce.map(x=>x.iv?x.iv*100:null), borderColor:'#1a7a3c', borderWidth:2, pointRadius:3, fill:false},
      {label:'PE IV%', data:pe.map(x=>x.iv?x.iv*100:null), borderColor:'#c0392b', borderWidth:2, pointRadius:3, fill:false},
    ]},
    options:{animation:false, plugins:{legend:{labels:{color:'#111'}}},
      scales:{x:{ticks:{color:'#555',maxTicksLimit:10}}, y:{ticks:{color:'#555'}, title:{display:true,text:'IV %'}}}}
  });
}

// ── Historical ────────────────────────────────────────────────────────────────
let histChart=null, normChart=null;
async function fetchHistorical() {
  const sym = document.getElementById('h-asset').value;
  const interval = document.getElementById('h-interval').value;
  const start = document.getElementById('h-start').value;
  const r = await fetch(`${API}/historical?symbol=${sym}&interval=${interval}&start=${encodeURIComponent(start)}&fields=open,high,low,close,tick_volume`);
  const d = await r.json();
  try {
    const vals = d.result[0].values[0][sym];
    const closes = vals.close || [];
    const labels = closes.map(p => {
      const dt = new Date(p.ts/1e6);
      return interval==='1d'||interval==='1w' ? dt.toLocaleDateString('en-IN') :
        dt.toLocaleString('en-IN',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    });
    const data = closes.map(p => (p.v/100).toFixed(2));
    const ctx = document.getElementById('hist-chart').getContext('2d');
    if (histChart) histChart.destroy();
    histChart = new Chart(ctx, {
      type:'line',
      data:{labels, datasets:[{label:sym, data, borderColor:'#111', borderWidth:1.5, pointRadius:0, fill:false}]},
      options:{animation:false, plugins:{legend:{labels:{color:'#111'}}},
        scales:{x:{ticks:{color:'#555',maxTicksLimit:10},grid:{color:'#f0f0f0'}},
                y:{ticks:{color:'#555'},grid:{color:'#f0f0f0'}}}}
    });
  } catch(e) { console.error(e); }
}

async function fetchNormReturns() {
  const assets = ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY','SENSEX','BANKEX'];
  const colors = {'NIFTY':'#111','BANKNIFTY':'#c0392b','FINNIFTY':'#1a7a3c','MIDCPNIFTY':'#2471a3','SENSEX':'#7d3c98','BANKEX':'#d35400'};
  const datasets = [];
  for (const asset of assets) {
    try {
      const exc = ['SENSEX','BANKEX'].includes(asset)?'BSE':'NSE';
      const r = await fetch(`${API}/historical?symbol=${asset}&exchange=${exc}&interval=1d&start=2025-01-01T03:45:00.000Z&fields=close`);
      const d = await r.json();
      const closes = d.result[0].values[0][asset]?.close || [];
      if (!closes.length) continue;
      const base = closes[0].v;
      datasets.push({
        label:asset, data:closes.map(p=>((p.v/base)*100).toFixed(2)),
        borderColor:colors[asset], borderWidth:1.5, pointRadius:0, fill:false,
      });
    } catch(e) {}
  }
  if (!datasets.length) return;
  const labels = Array.from({length: Math.max(...datasets.map(d=>d.data.length))}, (_,i)=>i+1);
  const ctx = document.getElementById('norm-chart').getContext('2d');
  if (normChart) normChart.destroy();
  normChart = new Chart(ctx, {
    type:'line', data:{labels, datasets},
    options:{animation:false, plugins:{legend:{labels:{color:'#111',font:{size:10}}}},
      scales:{x:{ticks:{color:'#555',maxTicksLimit:10},grid:{color:'#f0f0f0'}},
              y:{ticks:{color:'#555'},grid:{color:'#f0f0f0'}}}}
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectWS();
loadDashboard();
setInterval(loadDashboard, 30000);

// ── Market status check ───────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const mktBadge = document.getElementById('mkt-badge');
    const status = d.market_status || 'unknown';
    mktBadge.textContent = status === 'open' ? 'MARKET OPEN' :
                           status === 'post_market' ? 'POST-MARKET' : 'MARKET CLOSED';
    mktBadge.className = 'hbadge' + (status === 'open' ? ' live' : '');
    mktBadge.title = d.note || '';
    document.getElementById('hstatus').textContent = d.ist_time || '';
  } catch(e) {}
}
setInterval(checkHealth, 60000);
checkHealth();
