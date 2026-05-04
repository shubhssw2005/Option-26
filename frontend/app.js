'use strict'; // v2

// ── API base ──────────────────────────────────────────────────────────────────
const API = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : window.location.origin;

const TIMEOUT = 60000;
let charts = {}, mktStatus = 'unknown';

// ── Utilities ─────────────────────────────────────────────────────────────────
const fmt  = (v, d) => { d = (d !== undefined) ? d : 2; return (v != null && !isNaN(v)) ? Number(v).toFixed(d) : '\u2014'; };
const fmtN = v => (v != null) ? Number(v).toLocaleString('en-IN') : '\u2014';
const fmtP = (v, d) => { d = (d !== undefined) ? d : 1; return (v != null && !isNaN(v)) ? Number(v).toFixed(d) + '%' : '\u2014'; };
const clr  = v => v > 0 ? 'up' : v < 0 ? 'down' : 'neutral';
const sign = v => v > 0 ? '+' : '';

async function apiFetch(path, retries) {
  retries = (retries !== undefined) ? retries : 2;
  for (let i = 0; i <= retries; i++) {
    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), TIMEOUT);
    try {
      const r = await fetch(API + path, { signal: ctrl.signal });
      clearTimeout(tid);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return await r.json();
    } catch (e) {
      clearTimeout(tid);
      if (i === retries) throw e;
      await new Promise(res => setTimeout(res, 3000));
    }
  }
}

function loading(id, msg) {
  if (msg === undefined) msg = 'Loading\u2026';
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (el) el.innerHTML = '<div class="loading">' + msg + '</div>';
}

function errMsg(id, msg) {
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (el) el.innerHTML = '<div class="error-msg">\u26a0 ' + msg + '</div>';
}

function histNote() {
  return mktStatus !== 'open'
    ? '<span style="font-size:9px;color:#886600;margin-left:8px">\u2139 Using historical data</span>'
    : '';
}

// ── Navigation ────────────────────────────────────────────────────────────────
function show(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.getAttribute('onclick') && n.getAttribute('onclick').indexOf("'" + page + "'") !== -1) {
      n.classList.add('active');
    }
  });
  // Auto-load data when switching pages
  const loaders = {
    volatility: loadVol,
    greeks: loadGreeks,
    history: loadHistory,
    signals: loadSignals,
    strategy: loadStrategy
  };
  if (loaders[page]) loaders[page]();
}

// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(function() {
  const el = document.getElementById('ist-time');
  if (el) el.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false, timeZone: 'Asia/Kolkata' }) + ' IST';
}, 1000);

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const d = await apiFetch('/health');
    mktStatus = d.market_status || 'unknown';

    const authDot   = document.getElementById('auth-dot');
    const authLabel = document.getElementById('auth-label');
    const wsDot     = document.getElementById('ws-dot');
    const wsLabel   = document.getElementById('ws-label');
    const mktEl     = document.getElementById('mkt-status');
    const liveBadge = document.getElementById('live-badge');

    if (authDot)   authDot.className   = 'status-dot' + (d.authenticated ? ' live' : ' warn');
    if (authLabel) authLabel.textContent = d.authenticated ? 'Authenticated' : 'Auth Failed';

    if (mktEl) {
      if (mktStatus === 'open') {
        mktEl.textContent = '\u25cf MARKET OPEN';
        mktEl.style.color = '#00ff88';
      } else if (mktStatus === 'post_market') {
        mktEl.textContent = '\u25d0 POST-MARKET';
        mktEl.style.color = '#ffaa00';
      } else {
        mktEl.textContent = '\u25cb MARKET CLOSED';
        mktEl.style.color = '#888';
      }
    }

    if (liveBadge) {
      liveBadge.textContent = mktStatus === 'open' ? '\u25cf LIVE' : 'MARKET CLOSED';
      liveBadge.style.color = mktStatus === 'open' ? '#00aa44' : '#888';
    }

    // WS dot — show as connected if health OK
    if (wsDot)   wsDot.className   = 'status-dot' + (d.authenticated ? ' live' : '');
    if (wsLabel) wsLabel.textContent = d.authenticated ? 'WS Live' : 'WS Off';

    if (d.ist_time) {
      const el = document.getElementById('ist-time');
      if (el) el.textContent = d.ist_time + ' IST';
    }
  } catch (e) {
    const authLabel = document.getElementById('auth-label');
    if (authLabel) authLabel.textContent = 'Offline';
  }
}
setInterval(checkHealth, 30000);

// ── Regime helper ─────────────────────────────────────────────────────────────
function determineRegime(ret5, ret20, rv5, rv20, garch1d) {
  const volExpanding = rv5 > rv20 * 1.3;
  const strongBear   = ret5 < -2 && ret20 < -5;
  const strongBull   = ret5 > 2  && ret20 > 5;
  const ranging      = Math.abs(ret5) < 1 && Math.abs(ret20) < 3;
  const highVol      = garch1d > 2.0;

  if (strongBear && volExpanding) return { label: 'BEARISH + VOL EXPANSION', css: 'bearish',   description: 'Sustained selling with rising volatility. Protective strategies, bear spreads.' };
  if (strongBear)                 return { label: 'BEARISH TREND',           css: 'bearish',   description: 'Downtrend in place. Bear spreads, protective puts, avoid naked calls.' };
  if (strongBull && volExpanding) return { label: 'BULLISH + VOL EXPANSION', css: 'bullish',   description: 'Rally with rising vol. Bull spreads, covered calls to monetise premium.' };
  if (strongBull)                 return { label: 'BULLISH TREND',           css: 'bullish',   description: 'Uptrend. Bull call spreads, covered calls, cash-secured puts.' };
  if (ranging && highVol)         return { label: 'RANGING + HIGH VOL',      css: 'volatile',  description: 'Sideways with elevated vol. Sell premium: straddles, strangles, iron condors.' };
  if (ranging)                    return { label: 'RANGING / CONSOLIDATION', css: 'ranging',   description: 'Low directional bias. Income strategies: covered calls, iron condors.' };
  if (volExpanding)               return { label: 'VOL EXPANSION',           css: 'volatile',  description: 'Vol rising without clear direction. Long straddles, protective options.' };
  return                                 { label: 'UNCERTAIN',               css: 'uncertain', description: 'Mixed signals. Use defined-risk strategies only.' };
}

// ── Overview ──────────────────────────────────────────────────────────────────
async function loadOverview() {
  const asset = (document.getElementById('ov-asset') || {}).value || 'NIFTY';
  const updEl = document.getElementById('ov-updated');
  if (updEl) updEl.textContent = 'Loading\u2026' + histNote();

  // Fetch vol forecast (always available — uses cached/historical data)
  let vf = {};
  try {
    vf = await apiFetch('/vol-forecast?symbol=' + asset);
  } catch (e) { /* continue with defaults */ }

  const garch1d  = vf.ensemble_vol_1d  || vf.garch_vol_1d  || 0;
  const garchAnn = vf.ensemble_vol_ann || vf.garch_vol_ann  || 0;
  const rv20     = vf.realized_vol_20d || 0;
  const rv5      = vf.realized_vol_5d  || 0;

  // Fetch strategy recommendation for spot price and returns
  let strat = {};
  try {
    strat = await apiFetch('/strategy-recommend?asset=' + asset + '&dte=7');
  } catch (e) { /* continue */ }

  const spot       = strat.spot || 0;
  const ret5       = strat.spot_change_pct || 0;
  const ret20      = 0; // not directly available from this endpoint
  const ivLevel    = strat.iv_level || 0;
  const ivRank     = strat.iv_rank  || 0;

  // Determine regime
  const regime = determineRegime(ret5, ret20, rv5, rv20, garch1d);

  // Render regime bar
  const regimeBar = document.getElementById('regime-bar');
  if (regimeBar) {
    regimeBar.innerHTML =
      '<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;background:#fafafa;border:1px solid #e8e8e8;border-radius:2px">' +
        '<span class="regime regime-' + regime.css + '">' + regime.label + '</span>' +
        '<span style="font-size:11px;color:#555">' + regime.description + '</span>' +
        '<span style="margin-left:auto;font-size:10px;color:#888">' + asset + ' \u00b7 ' + new Date().toLocaleDateString('en-IN') + '</span>' +
        (mktStatus !== 'open' ? '<span style="font-size:9px;color:#886600">\u2139 Historical data</span>' : '') +
      '</div>';
  }

  // Render KPI row
  const kpiRow = document.getElementById('kpi-row');
  if (kpiRow) {
    const kpis = [
      { label: 'Spot Price',    value: spot ? spot.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '\u2014', sub: 'Last close', cls: '' },
      { label: '5d Return',     value: fmtP(ret5),    sub: 'Price change',         cls: ret5 < 0 ? 'alert' : ret5 > 0 ? 'positive' : '' },
      { label: 'GARCH 1d Vol',  value: fmtP(garch1d), sub: 'Ann: ' + fmtP(garchAnn), cls: garch1d > 2.5 ? 'alert' : '' },
      { label: 'Realized 20d',  value: fmtP(rv20),    sub: '5d: ' + fmtP(rv5),    cls: rv5 > rv20 * 1.5 ? 'alert' : '' },
      { label: 'IV Level',      value: fmtP(ivLevel), sub: 'IV Rank: ' + fmtP(ivRank), cls: ivLevel > 25 ? 'alert' : '' },
    ];
    kpiRow.innerHTML = kpis.map(function(k) {
      return '<div class="kpi ' + k.cls + '">' +
        '<div class="kpi-label">' + k.label + '</div>' +
        '<div class="kpi-value ' + (k.cls === 'alert' ? 'down' : k.cls === 'positive' ? 'up' : '') + '">' + k.value + '</div>' +
        '<div class="kpi-sub">' + k.sub + '</div>' +
      '</div>';
    }).join('');
  }

  // Load strategy cards for overview
  loadStrategyForOverview(asset, regime, strat);

  // Load vol regime panel
  loadVolRegimes();

  // Load live index
  loadLiveIndex();

  if (updEl) updEl.textContent = 'Updated ' + new Date().toLocaleTimeString('en-IN', { hour12: false }) + histNote();
}

async function loadStrategyForOverview(asset, regime, stratData) {
  loading('strat-cards');
  const regimeLabel = document.getElementById('strat-regime-label');
  if (regimeLabel) regimeLabel.textContent = regime.label;

  try {
    const d = stratData && stratData.top_recommendations ? stratData : await apiFetch('/strategy-recommend?asset=' + asset + '&dte=7');
    const top = d.top_recommendations || [];
    const el = document.getElementById('strat-cards');
    if (!el) return;

    if (!top.length) {
      el.innerHTML = '<div style="color:#888;font-size:11px">No recommendations available</div>';
      return;
    }

    el.innerHTML = top.slice(0, 4).map(function(s, i) {
      const score  = s.score || 0;
      const tf     = s.trend_fit || 'neutral';
      const tagCls = tf.indexOf('bull') !== -1 ? 'tag-bull' : tf.indexOf('bear') !== -1 ? 'tag-bear' : 'tag-neutral';
      const catCls = s.category === 'income' ? 'tag-income' : s.category === 'volatility' ? 'tag-vol' : 'tag-neutral';
      return '<div class="strat-card ' + (i === 0 ? 'top-pick' : '') + '">' +
        '<div class="strat-name">' + (s.name || s.strategy_key || 'Strategy') + '</div>' +
        '<div class="strat-score-bar"><div class="strat-score-fill" style="width:' + score + '%"></div></div>' +
        '<div style="font-size:9px;color:#888;margin-bottom:6px">Confidence: ' + score.toFixed(0) + '/100</div>' +
        '<div class="strat-tags">' +
          '<span class="tag ' + tagCls + '">' + tf + '</span>' +
          '<span class="tag ' + catCls + '">' + (s.category || '\u2014') + '</span>' +
        '</div>' +
        '<div class="strat-detail">' + (s.description || s.best_for || '\u2014') + '</div>' +
      '</div>';
    }).join('');
  } catch (e) {
    errMsg('strat-cards', e.message);
  }
}

async function loadVolRegimes() {
  const el = document.getElementById('vol-regime-panel');
  if (!el) return;
  loading(el);
  const assets = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'];
  const results = await Promise.allSettled(assets.map(function(a) {
    return apiFetch('/vol-forecast?symbol=' + a).then(function(d) { return [a, d]; });
  }));
  el.innerHTML = results.map(function(r) {
    if (r.status !== 'fulfilled') return '';
    const a  = r.value[0];
    const vf = r.value[1];
    const v1d  = vf.ensemble_vol_1d || vf.garch_vol_1d || 0;
    const rv20 = vf.realized_vol_20d || 0;
    const pct  = Math.min((v1d / 4) * 100, 100);
    const cls  = v1d > 2.5 ? 'vg-high' : v1d > 1.5 ? 'vg-mid' : 'vg-low';
    const lbl  = v1d > 2.5 ? 'HIGH' : v1d > 1.5 ? 'ELEVATED' : 'NORMAL';
    return '<div style="margin-bottom:8px">' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:2px">' +
        '<span style="font-weight:600;font-size:11px">' + a + '</span>' +
        '<span style="font-size:10px;color:#888">' + lbl + ' \u00b7 GARCH ' + fmtP(v1d) + ' \u00b7 RV20 ' + fmtP(rv20) + '</span>' +
      '</div>' +
      '<div class="vol-gauge"><div class="vol-gauge-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
    '</div>';
  }).join('');
}

async function loadLiveIndex() {
  const el = document.getElementById('live-index-panel');
  if (!el) return;
  loading(el);
  try {
    const data = await apiFetch('/live/index');
    if (!Array.isArray(data) || !data.length) {
      el.innerHTML = '<div style="color:#888;font-size:11px">No index data available</div>';
      return;
    }
    el.innerHTML = data.map(function(idx) {
      const close  = idx.close  || 0;
      const chgPct = idx.changepercent || idx.change_percent || 0;
      const chg    = idx.change || 0;
      return '<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f5f5f5">' +
        '<span style="font-weight:600">' + (idx.indexname || idx.name || '\u2014') + '</span>' +
        '<span>' + (close ? close.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '\u2014') +
          ' <span class="' + clr(chgPct) + '">' + (chgPct >= 0 ? '+' : '') + Number(chgPct).toFixed(2) + '%</span>' +
        '</span>' +
      '</div>';
    }).join('');
  } catch (e) {
    el.innerHTML = '<div style="color:#888;font-size:11px">Index data unavailable</div>';
  }
}

// ── Strategy Engine ───────────────────────────────────────────────────────────
async function loadStrategy() {
  const asset = (document.getElementById('st-asset') || {}).value || 'NIFTY';
  const dte   = (document.getElementById('st-dte')   || {}).value || '7';
  loading('st-context');
  loading('st-reasoning');
  loading('st-cards');

  // Fetch vol forecast for context
  let vf = {};
  try { vf = await apiFetch('/vol-forecast?symbol=' + asset); } catch (e) {}

  const garch1d = vf.ensemble_vol_1d || vf.garch_vol_1d || 0;
  const rv20    = vf.realized_vol_20d || 0;
  const rv5     = vf.realized_vol_5d  || 0;

  // Fetch strategy recommendation
  let d = {};
  try {
    d = await apiFetch('/strategy-recommend?asset=' + asset + '&dte=' + dte);
  } catch (e) {
    errMsg('st-cards', e.message);
    return;
  }

  const spot    = d.spot || 0;
  const ret5    = d.spot_change_pct || 0;
  const ivLevel = d.iv_level || 0;
  const ivRank  = d.iv_rank  || 0;
  const regime  = d.regime   || 'unknown';
  const regDesc = d.regime_description || '';
  const cond    = d.condition_analysis || {};
  const top     = d.top_recommendations || [];

  // Market context bar
  const ctxEl = document.getElementById('st-context');
  if (ctxEl) {
    const regCss = regime.toLowerCase().indexOf('bull') !== -1 ? 'bullish' :
                   regime.toLowerCase().indexOf('bear') !== -1 ? 'bearish' :
                   regime.toLowerCase().indexOf('vol')  !== -1 ? 'volatile' : 'uncertain';
    ctxEl.innerHTML =
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:8px 12px;background:#fafafa;border:1px solid #e8e8e8;border-radius:2px;margin-bottom:10px">' +
        '<span class="regime regime-' + regCss + '">' + regime.toUpperCase() + '</span>' +
        '<span style="font-size:11px">Spot: <strong>' + (spot ? spot.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '\u2014') + '</strong></span>' +
        '<span style="font-size:11px">5d: <strong class="' + clr(ret5) + '">' + sign(ret5) + fmtP(ret5) + '</strong></span>' +
        '<span style="font-size:11px">IV: <strong>' + fmtP(ivLevel) + '</strong></span>' +
        '<span style="font-size:11px">IV Rank: <strong>' + fmtP(ivRank) + '</strong></span>' +
        '<span style="font-size:11px">GARCH: <strong>' + fmtP(garch1d) + '</strong></span>' +
        '<span style="font-size:11px">RV20: <strong>' + fmtP(rv20) + '</strong></span>' +
        '<span style="font-size:11px">DTE: <strong>' + dte + 'd</strong></span>' +
        histNote() +
      '</div>';
  }

  // Reasoning panel
  const reasonEl = document.getElementById('st-reasoning');
  if (reasonEl) {
    const volPremium = garch1d > rv20
      ? 'GARCH forecasts higher vol than realized \u2014 vol likely to stay elevated'
      : 'GARCH below realized \u2014 vol may mean-revert lower';
    const volSkew = rv5 > rv20 * 1.3
      ? 'Short-term vol (5d) >> Long-term (20d) \u2014 vol spike in progress'
      : 'Vol term structure normal';

    reasonEl.innerHTML =
      '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:11px">' +
        '<div class="risk-panel">' +
          '<div style="font-weight:700;margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:.06em">Market Conditions</div>' +
          '<div class="risk-row"><span class="risk-label">Trend</span><span class="risk-value">' + (cond.trend || '\u2014') + '</span></div>' +
          '<div class="risk-row"><span class="risk-label">Vol Environment</span><span class="risk-value">' + (cond.volatility_environment || '\u2014') + '</span></div>' +
          '<div class="risk-row"><span class="risk-label">IV Status</span><span class="risk-value">' + (cond.iv_status || '\u2014') + '</span></div>' +
          '<div class="risk-row"><span class="risk-label">Outlook</span><span class="risk-value">' + (cond.overall_outlook || '\u2014') + '</span></div>' +
        '</div>' +
        '<div class="risk-panel">' +
          '<div style="font-weight:700;margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:.06em">Volatility Analysis</div>' +
          '<div class="risk-row"><span class="risk-label">GARCH 1d</span><span class="risk-value">' + fmtP(garch1d) + '</span></div>' +
          '<div class="risk-row"><span class="risk-label">Realized 20d</span><span class="risk-value">' + fmtP(rv20) + '</span></div>' +
          '<div class="risk-row"><span class="risk-label">Realized 5d</span><span class="risk-value ' + (rv5 > rv20 * 1.3 ? 'down' : '') + '">' + fmtP(rv5) + '</span></div>' +
          '<div style="margin-top:6px;color:#555;font-size:10px">' + volSkew + '</div>' +
        '</div>' +
        '<div class="risk-panel">' +
          '<div style="font-weight:700;margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:.06em">Strategy Logic</div>' +
          '<div style="color:#555;font-size:10px;line-height:1.7">' + (regDesc || regime) + '<br><br>' + volPremium + '</div>' +
        '</div>' +
      '</div>';
  }

  // Strategy cards
  const cardsEl = document.getElementById('st-cards');
  if (cardsEl) {
    if (!top.length) {
      cardsEl.innerHTML = '<div style="color:#888;font-size:11px">No strategy recommendations available</div>';
    } else {
      cardsEl.innerHTML = top.slice(0, 6).map(function(s, i) {
        const score  = s.score || 0;
        const tf     = s.trend_fit || 'neutral';
        const tagCls = tf.indexOf('bull') !== -1 ? 'tag-bull' : tf.indexOf('bear') !== -1 ? 'tag-bear' : 'tag-neutral';
        const catCls = s.category === 'income' ? 'tag-income' : s.category === 'volatility' ? 'tag-vol' : 'tag-neutral';
        const riskScore = s.risk_score || 3;
        const riskDots  = [1,2,3,4,5].map(function(n) {
          return '<div class="risk-dot ' + (n <= riskScore ? 'filled' : '') + '"></div>';
        }).join('');
        return '<div class="strat-card ' + (i === 0 ? 'top-pick' : '') + '" onclick="showStratDetail(\'' + (s.name || '') + '\',\'' + asset + '\')">' +
          '<div class="strat-name">' + (s.name || '\u2014') + '</div>' +
          '<div class="strat-score-bar"><div class="strat-score-fill" style="width:' + score + '%"></div></div>' +
          '<div style="font-size:9px;color:#888;margin-bottom:6px">Score: ' + score.toFixed(0) + '/100</div>' +
          '<div class="strat-tags">' +
            '<span class="tag ' + tagCls + '">' + tf + '</span>' +
            '<span class="tag ' + catCls + '">' + (s.category || '\u2014') + '</span>' +
          '</div>' +
          '<div class="strat-detail" style="margin-bottom:6px">' + (s.description || '\u2014') + '</div>' +
          '<div class="strat-detail"><strong>Risk:</strong> ' + (s.risk_profile || '\u2014') + '</div>' +
          '<div class="strat-risk" style="margin-top:6px">' +
            '<span style="font-size:9px;color:#888;margin-right:4px">RISK</span>' + riskDots +
          '</div>' +
        '</div>';
      }).join('');
    }
  }
}

function showStratDetail(name, asset) {
  const panel = document.getElementById('st-detail');
  if (panel) panel.style.display = 'block';
  const titleEl = document.getElementById('st-detail-title');
  const bodyEl  = document.getElementById('st-detail-body');
  if (titleEl) titleEl.textContent = name || 'Strategy Detail';
  if (bodyEl) {
    bodyEl.innerHTML =
      '<div style="font-size:11px;color:#555;line-height:1.8;padding:10px 12px;background:#fafafa;border:1px solid #e8e8e8;border-radius:2px">' +
        '<strong>' + (name || 'Strategy') + '</strong> for <strong>' + asset + '</strong><br><br>' +
        'Review the strategy score and risk profile above. Set stop-loss at 50% of max loss. ' +
        'Exit if underlying moves against position by more than 1 ATR. Monitor delta daily.' +
      '</div>';
  }
  if (panel) panel.scrollIntoView({ behavior: 'smooth' });
}

// ── Volatility ────────────────────────────────────────────────────────────────
async function loadVol() {
  const asset = (document.getElementById('vol-asset') || {}).value || 'NIFTY';
  loading('vol-kpis');
  loading('vol-all-panel');
  loading('vol-interpretation');

  let vf = {};
  try {
    vf = await apiFetch('/vol-forecast?symbol=' + asset);
  } catch (e) {
    errMsg('vol-kpis', 'Failed to load: ' + e.message);
    return;
  }

  const v1d  = vf.ensemble_vol_1d  || vf.garch_vol_1d  || 0;
  const vann = vf.ensemble_vol_ann || vf.garch_vol_ann  || 0;
  const rv20 = vf.realized_vol_20d || 0;
  const rv5  = vf.realized_vol_5d  || 0;

  // KPI tiles
  const kpisEl = document.getElementById('vol-kpis');
  if (kpisEl) {
    const kpis = [
      { label: 'GARCH 1d',     value: fmtP(v1d),          sub: 'Ann: ' + fmtP(vann),          cls: v1d > 2.5 ? 'alert' : '' },
      { label: 'Realized 20d', value: fmtP(rv20),          sub: '5d: ' + fmtP(rv5),            cls: rv5 > rv20 * 1.3 ? 'alert' : '' },
      { label: 'Vol Premium',  value: (v1d && rv20) ? fmtP(v1d - rv20) : '\u2014', sub: v1d > rv20 ? 'GARCH > RV' : 'GARCH < RV', cls: '' },
      { label: 'Vol Regime',   value: v1d > 2.5 ? 'HIGH' : v1d > 1.5 ? 'ELEVATED' : 'NORMAL', sub: 'Based on GARCH', cls: v1d > 2.5 ? 'alert' : '' },
    ];
    kpisEl.innerHTML = kpis.map(function(k) {
      return '<div class="kpi ' + k.cls + '">' +
        '<div class="kpi-label">' + k.label + '</div>' +
        '<div class="kpi-value">' + k.value + '</div>' +
        '<div class="kpi-sub">' + k.sub + '</div>' +
      '</div>';
    }).join('');
  }

  // GARCH model comparison chart
  const models  = vf.models || {};
  const mNames  = Object.keys(models);
  const mVols   = mNames.map(function(m) { return models[m].vol_1d || 0; });
  const ctx1    = document.getElementById('garch-chart');
  if (ctx1 && mNames.length) {
    if (charts.garch) { charts.garch.destroy(); charts.garch = null; }
    charts.garch = new Chart(ctx1.getContext('2d'), {
      type: 'bar',
      data: {
        labels: mNames.map(function(n) { return n.toUpperCase(); }),
        datasets: [{ label: '1-Day Vol%', data: mVols, backgroundColor: '#111', borderRadius: 1 }]
      },
      options: {
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#888', font: { size: 10 } } },
          y: { ticks: { color: '#888', font: { size: 10 } }, title: { display: true, text: 'Vol %', color: '#888', font: { size: 10 } } }
        }
      }
    });
  } else if (ctx1 && !mNames.length) {
    // Show SARIMA forecast if no model breakdown
    const sarima = vf.sarima_forecast || [];
    if (sarima.length) {
      if (charts.garch) { charts.garch.destroy(); charts.garch = null; }
      charts.garch = new Chart(ctx1.getContext('2d'), {
        type: 'line',
        data: {
          labels: sarima.map(function(_, i) { return 'Day ' + (i + 1); }),
          datasets: [{ label: 'SARIMA Forecast', data: sarima, borderColor: '#111', borderWidth: 1.5, pointRadius: 2, fill: false }]
        },
        options: {
          animation: false,
          plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
          scales: {
            x: { ticks: { color: '#888', font: { size: 10 } } },
            y: { ticks: { color: '#888', font: { size: 10 } }, title: { display: true, text: 'Vol %', color: '#888', font: { size: 10 } } }
          }
        }
      });
    }
  }

  // All assets vol table
  const allAssets = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'];
  const rows = await Promise.allSettled(allAssets.map(function(a) {
    return apiFetch('/vol-forecast?symbol=' + a).then(function(d) { return [a, d]; });
  }));
  const allEl = document.getElementById('vol-all-panel');
  if (allEl) {
    allEl.innerHTML =
      '<table class="tbl"><thead><tr><th>Asset</th><th>GARCH 1d</th><th>Ann Vol</th><th>RV 20d</th><th>RV 5d</th><th>Regime</th></tr></thead><tbody>' +
      rows.map(function(r) {
        if (r.status !== 'fulfilled') return '';
        const a  = r.value[0];
        const v  = r.value[1];
        const v1 = v.ensemble_vol_1d || v.garch_vol_1d || 0;
        const va = v.ensemble_vol_ann || v.garch_vol_ann || 0;
        const rv = v.realized_vol_20d || 0;
        const r5 = v.realized_vol_5d  || 0;
        const reg = v1 > 2.5 ? 'HIGH' : v1 > 1.5 ? 'ELEVATED' : 'NORMAL';
        return '<tr>' +
          '<td><strong>' + a + '</strong></td>' +
          '<td class="' + (v1 > 2.5 ? 'down' : '') + '">' + fmtP(v1) + '</td>' +
          '<td>' + fmtP(va) + '</td>' +
          '<td>' + fmtP(rv) + '</td>' +
          '<td class="' + (r5 > rv * 1.3 ? 'down' : '') + '">' + fmtP(r5) + '</td>' +
          '<td>' + reg + '</td>' +
        '</tr>';
      }).join('') +
      '</tbody></table>';
  }

  // Interpretation
  const interpEl = document.getElementById('vol-interpretation');
  if (interpEl) {
    let interp;
    if (v1d > 2.5) {
      interp = '<strong>High volatility environment (' + fmtP(v1d) + ' daily).</strong> Options are expensive. Favour selling premium: short straddles, iron condors, covered calls. Avoid buying options unless for protection.';
    } else if (v1d > 1.5) {
      interp = '<strong>Elevated volatility (' + fmtP(v1d) + ' daily).</strong> Balanced environment. Both buying and selling strategies viable. Prefer defined-risk spreads.';
    } else {
      interp = '<strong>Low volatility (' + fmtP(v1d) + ' daily).</strong> Options are cheap. Favour buying: long straddles, long strangles, calendar spreads. Good time to buy protection.';
    }
    if (mktStatus !== 'open') {
      interp += ' <em style="color:#886600">(Based on historical/cached data)</em>';
    }
    interpEl.innerHTML = '<div style="font-size:11px;line-height:1.8;color:#333">' + interp + '</div>';
  }
}

// ── Option Chain ──────────────────────────────────────────────────────────────
async function loadChain() {
  const sym = (document.getElementById('oc-asset')    || {}).value || 'NIFTY';
  const exc = (document.getElementById('oc-exchange') || {}).value || 'NSE';
  const exp = (document.getElementById('oc-expiry')   || {}).value || '';

  if (!exp) {
    alert('Enter expiry date in YYYYMMDD format');
    return;
  }

  const ceEl = document.getElementById('ce-table');
  const peEl = document.getElementById('pe-table');
  if (ceEl) ceEl.innerHTML = '<tr><td colspan="9" class="loading">Loading\u2026</td></tr>';
  if (peEl) peEl.innerHTML = '<tr><td colspan="9" class="loading">Loading\u2026</td></tr>';

  try {
    const d     = await apiFetch('/option-chain?instrument=' + sym + '&exchange=' + exc + '&expiry=' + exp);
    const chain = d.chain || {};
    const atm   = chain.atm ? chain.atm / 100 : null;
    const spot  = chain.cp  ? chain.cp  / 100 : null;

    const metaEl = document.getElementById('chain-meta');
    if (metaEl) {
      metaEl.textContent = atm
        ? 'ATM: ' + atm.toLocaleString('en-IN') + ' | Spot: ' + (spot ? spot.toLocaleString('en-IN') : '\u2014') + ' | Expiry: ' + exp
        : 'Expiry: ' + exp;
    }

    renderChainTable('ce-table', chain.ce || [], atm, 'ce-row');
    renderChainTable('pe-table', chain.pe || [], atm, 'pe-row');
  } catch (e) {
    if (ceEl) ceEl.innerHTML = '<tr><td colspan="9" class="error-msg">\u26a0 ' + e.message + '</td></tr>';
  }
}

function renderChainTable(id, items, atm, rowCls) {
  const el = document.getElementById(id);
  if (!el) return;
  const sorted = items.slice().sort(function(a, b) { return (a.sp || 0) - (b.sp || 0); }).slice(0, 30);
  if (!sorted.length) {
    el.innerHTML = '<tr><td colspan="9" style="color:#888;padding:8px">No data</td></tr>';
    return;
  }
  el.innerHTML = sorted.map(function(i) {
    const strike   = i.sp != null ? (i.sp / 100).toFixed(0) : '\u2014';
    const isAtm    = atm && i.sp != null && Math.abs(i.sp / 100 - atm) < 50;
    const ltp      = i.ltp  != null ? (i.ltp  / 100).toFixed(2) : '\u2014';
    const iv       = i.iv   != null ? (i.iv   * 100).toFixed(1) : '\u2014';
    const ivColor  = i.iv > 0.3 ? 'down' : i.iv > 0.2 ? 'neutral' : 'up';
    return '<tr class="' + (isAtm ? 'atm' : rowCls) + '">' +
      '<td><strong>' + Number(strike).toLocaleString('en-IN') + '</strong></td>' +
      '<td>' + ltp + '</td>' +
      '<td class="' + ivColor + '">' + iv + '</td>' +
      '<td class="' + clr(i.delta) + '">' + fmt(i.delta, 3) + '</td>' +
      '<td>' + fmt(i.gamma, 5) + '</td>' +
      '<td class="' + clr(i.theta) + '">' + fmt(i.theta, 2) + '</td>' +
      '<td>' + fmt(i.vega, 2) + '</td>' +
      '<td>' + fmtN(i.oi) + '</td>' +
      '<td>' + fmtN(i.volume) + '</td>' +
    '</tr>';
  }).join('');
}

// ── ML Signals ────────────────────────────────────────────────────────────────
async function loadSignals() {
  const asset = (document.getElementById('sig-asset') || {}).value || 'NIFTY';
  const ceEl  = document.getElementById('ce-signals');
  const peEl  = document.getElementById('pe-signals');
  if (ceEl) ceEl.innerHTML = '<tr><td colspan="7" class="loading">Loading\u2026</td></tr>';
  if (peEl) peEl.innerHTML = '<tr><td colspan="7" class="loading">Loading\u2026</td></tr>';

  try {
    const d = await apiFetch('/signals?asset=' + asset);
    renderSignalTable('ce-signals', d.CE || [], '#00aa44');
    renderSignalTable('pe-signals', d.PE || [], '#cc2200');
  } catch (e) {
    if (ceEl) ceEl.innerHTML = '<tr><td colspan="7" class="error-msg">\u26a0 ' + e.message + '</td></tr>';
  }
}

function renderSignalTable(id, rows, color) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!rows.length) {
    el.innerHTML = '<tr><td colspan="7" style="color:#888;padding:8px">No signals \u2014 run collect_data.py during market hours</td></tr>';
    return;
  }
  el.innerHTML = rows.map(function(r) {
    const score  = (r.signal_score || 0) * 100;
    const action = score > 70
      ? '<span class="signal-high">BUY</span>'
      : score > 55
        ? 'WATCH'
        : '<span style="color:#888">SKIP</span>';
    const strike = r.strike ? (r.strike / 100).toLocaleString('en-IN') : '\u2014';
    const barW   = Math.round(score * 0.5);
    return '<tr>' +
      '<td>' + strike + '</td>' +
      '<td>' + (r.ltp != null ? r.ltp.toFixed(2) : '\u2014') + '</td>' +
      '<td>' + (r.iv  != null ? (r.iv * 100).toFixed(1) + '%' : '\u2014') + '</td>' +
      '<td class="' + clr(r.delta) + '">' + fmt(r.delta, 3) + '</td>' +
      '<td>' + fmtN(r.oi) + '</td>' +
      '<td><div style="display:flex;align-items:center;gap:5px"><div style="width:' + barW + 'px;height:4px;background:' + color + ';border-radius:2px;min-width:2px"></div>' + score.toFixed(0) + '</div></td>' +
      '<td>' + action + '</td>' +
    '</tr>';
  }).join('');
}

async function loadAllSignals() {
  const panel = document.getElementById('all-signals-panel');
  if (panel) panel.style.display = 'block';
  loading('all-signals-body');
  try {
    const d = await apiFetch('/signals/all');
    const bodyEl = document.getElementById('all-signals-body');
    if (!bodyEl) return;
    bodyEl.innerHTML =
      '<table class="tbl"><thead><tr><th>Asset</th><th>Top CE Strike</th><th>CE Score</th><th>Top PE Strike</th><th>PE Score</th><th>Signal</th></tr></thead><tbody>' +
      Object.entries(d).map(function(entry) {
        const a    = entry[0];
        const sigs = entry[1];
        if (sigs.error) return '<tr><td>' + a + '</td><td colspan="5" style="color:#888">' + sigs.error + '</td></tr>';
        const ce      = (sigs.CE || [])[0];
        const pe      = (sigs.PE || [])[0];
        const ceScore = (ce ? (ce.signal_score || 0) : 0) * 100;
        const peScore = (pe ? (pe.signal_score || 0) : 0) * 100;
        const signal  = ceScore > 70
          ? '<span class="up">CALL BUY</span>'
          : peScore > 70
            ? '<span class="down">PUT BUY</span>'
            : '<span style="color:#888">NEUTRAL</span>';
        return '<tr>' +
          '<td><strong>' + a + '</strong></td>' +
          '<td>' + (ce && ce.strike ? (ce.strike / 100).toLocaleString('en-IN') : '\u2014') + '</td>' +
          '<td>' + ceScore.toFixed(0) + '</td>' +
          '<td>' + (pe && pe.strike ? (pe.strike / 100).toLocaleString('en-IN') : '\u2014') + '</td>' +
          '<td>' + peScore.toFixed(0) + '</td>' +
          '<td>' + signal + '</td>' +
        '</tr>';
      }).join('') +
      '</tbody></table>';
  } catch (e) {
    errMsg('all-signals-body', e.message);
  }
}

// ── Greeks / IV Surface ───────────────────────────────────────────────────────
async function loadGreeks() {
  const asset = (document.getElementById('gr-asset') || {}).value || 'NIFTY';
  try {
    const d = await apiFetch('/iv-surface?asset=' + asset);
    if (!Array.isArray(d) || !d.length) {
      ['delta-chart', 'gamma-chart', 'theta-chart', 'iv-chart'].forEach(function(id) {
        const el = document.getElementById(id);
        if (el && el.parentElement) el.parentElement.innerHTML = '<div style="color:#888;font-size:11px;padding:12px">No data available</div>';
      });
      return;
    }

    const ce     = d.filter(function(x) { return x.option_type === 'CE'; }).sort(function(a, b) { return a.strike - b.strike; });
    const pe     = d.filter(function(x) { return x.option_type === 'PE'; }).sort(function(a, b) { return a.strike - b.strike; });
    const labels = Array.from(new Set(d.map(function(x) { return (x.strike / 100).toFixed(0); })));

    function getArr(arr, field) {
      return arr.map(function(x) { return x[field] != null ? x[field] : null; });
    }

    function mkChart(id, ceData, peData, label, yLabel) {
      const canvas = document.getElementById(id);
      if (!canvas) return;
      if (charts[id]) { charts[id].destroy(); charts[id] = null; }
      charts[id] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            { label: 'CE ' + label, data: ceData, borderColor: '#00aa44', borderWidth: 1.5, pointRadius: 1, fill: false },
            { label: 'PE ' + label, data: peData, borderColor: '#cc2200', borderWidth: 1.5, pointRadius: 1, fill: false }
          ]
        },
        options: {
          animation: false,
          plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
          scales: {
            x: { ticks: { color: '#888', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#f5f5f5' } },
            y: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#f5f5f5' }, title: { display: true, text: yLabel, color: '#888', font: { size: 10 } } }
          }
        }
      });
    }

    // Note: HTML uses id="iv-chart" (not iv-smile-chart)
    mkChart('iv-chart',    ce.map(function(x) { return x.iv ? x.iv * 100 : null; }), pe.map(function(x) { return x.iv ? x.iv * 100 : null; }), 'IV', 'IV %');
    mkChart('delta-chart', getArr(ce, 'delta'), getArr(pe, 'delta'), 'Delta', 'Delta');
    mkChart('gamma-chart', getArr(ce, 'gamma'), getArr(pe, 'gamma'), 'Gamma', 'Gamma');
    mkChart('theta-chart', getArr(ce, 'theta'), getArr(pe, 'theta'), 'Theta', 'Theta');
  } catch (e) {
    console.error('loadGreeks error:', e);
  }
}

// ── Historical ────────────────────────────────────────────────────────────────
async function loadHistory() {
  const sym      = (document.getElementById('h-asset')    || {}).value || 'NIFTY';
  const interval = (document.getElementById('h-interval') || {}).value || '1d';
  const start    = (document.getElementById('h-start')    || {}).value || '2025-01-01T03:45:00.000Z';

  try {
    const d = await apiFetch('/historical?symbol=' + sym + '&interval=' + interval + '&start=' + encodeURIComponent(start) + '&fields=open,high,low,close,tick_volume');
    const closes = (d.result && d.result[0] && d.result[0].values && d.result[0].values[0] && d.result[0].values[0][sym] && d.result[0].values[0][sym].close) || [];

    if (!closes.length) {
      const histCanvas = document.getElementById('hist-chart');
      if (histCanvas && histCanvas.parentElement) {
        histCanvas.parentElement.innerHTML = '<div style="color:#888;font-size:11px;padding:12px">No historical data available</div>';
      }
      return;
    }

    const labels = closes.map(function(p) {
      const dt = new Date((p.ts || p.timestamp) / 1e6);
      return (interval === '1d' || interval === '1w')
        ? dt.toLocaleDateString('en-IN')
        : dt.toLocaleString('en-IN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    });
    const data = closes.map(function(p) { return ((p.v || p.value) / 100).toFixed(2); });

    const histCanvas = document.getElementById('hist-chart');
    if (histCanvas) {
      if (charts.hist) { charts.hist.destroy(); charts.hist = null; }
      charts.hist = new Chart(histCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{ label: sym, data: data, borderColor: '#111', borderWidth: 1.5, pointRadius: 0, fill: false }]
        },
        options: {
          animation: false,
          plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
          scales: {
            x: { ticks: { color: '#888', maxTicksLimit: 12, font: { size: 10 } }, grid: { color: '#f5f5f5' } },
            y: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#f5f5f5' } }
          }
        }
      });
    }
  } catch (e) {
    console.error('loadHistory error:', e);
  }

  loadNormReturns();
}

async function loadNormReturns() {
  const assets   = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX'];
  const colors   = { NIFTY: '#111', BANKNIFTY: '#cc2200', FINNIFTY: '#00aa44', MIDCPNIFTY: '#0066cc', SENSEX: '#886600', BANKEX: '#cc6600' };
  const datasets = [];

  for (let i = 0; i < assets.length; i++) {
    const a = assets[i];
    try {
      const exc = (a === 'SENSEX' || a === 'BANKEX') ? 'BSE' : 'NSE';
      const d   = await apiFetch('/historical?symbol=' + a + '&exchange=' + exc + '&interval=1d&start=2025-01-01T03:45:00.000Z&fields=close');
      const closes = (d.result && d.result[0] && d.result[0].values && d.result[0].values[0] && d.result[0].values[0][a] && d.result[0].values[0][a].close) || [];
      if (!closes.length) continue;
      const base = closes[0].v || closes[0].value;
      datasets.push({
        label: a,
        data: closes.map(function(p) { return (((p.v || p.value) / base) * 100).toFixed(2); }),
        borderColor: colors[a],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      });
    } catch (e) { /* skip */ }
  }

  if (!datasets.length) return;
  const normCanvas = document.getElementById('norm-chart');
  if (!normCanvas) return;
  if (charts.norm) { charts.norm.destroy(); charts.norm = null; }
  const maxLen = Math.max.apply(null, datasets.map(function(d) { return d.data.length; }));
  charts.norm = new Chart(normCanvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: Array.from({ length: maxLen }, function(_, i) { return i + 1; }),
      datasets: datasets
    },
    options: {
      animation: false,
      plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: '#888', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#f5f5f5' } },
        y: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#f5f5f5' } }
      }
    }
  });
}

// ── Auto-refresh when market is open ─────────────────────────────────────────
setInterval(function() {
  if (mktStatus === 'open') {
    const activePage = document.querySelector('.page.active');
    if (!activePage) return;
    const page = activePage.id.replace('page-', '');
    if (page === 'overview') loadOverview();
    if (page === 'signals')  loadSignals();
    if (page === 'greeks')   loadGreeks();
  }
}, 60000);

// ── WebSocket (live data when market open) ────────────────────────────────────
function connectWS() {
  const wsDot   = document.getElementById('ws-dot');
  const wsLabel = document.getElementById('ws-label');
  try {
    const wsUrl = (API.startsWith('https') ? 'wss' : 'ws') + '://' + window.location.host + '/ws/live';
    const ws = new WebSocket(wsUrl);
    ws.onopen = function() {
      if (wsDot)   wsDot.className    = 'status-dot live';
      if (wsLabel) wsLabel.textContent = 'WS Live';
    };
    ws.onclose = function() {
      if (wsDot)   wsDot.className    = 'status-dot';
      if (wsLabel) wsLabel.textContent = 'WS Off';
      setTimeout(connectWS, 5000);
    };
    ws.onerror = function() { ws.close(); };
    ws.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.indexes && Array.isArray(d.indexes) && d.indexes.length) {
          var el = document.getElementById('live-index-panel');
          if (el) renderLiveIndex(el, d.indexes);
        }
      } catch(err) {}
    };
  } catch(e) {
    if (wsDot)   wsDot.className    = 'status-dot warn';
    if (wsLabel) wsLabel.textContent = 'WS N/A';
  }
}

function renderLiveIndex(el, data) {
  el.innerHTML = data.map(function(idx) {
    var close  = idx.close || (idx.index_value ? idx.index_value / 100 : 0);
    var chg    = idx.change || 0;
    var chgPct = idx.changepercent || idx.changeprecent || 0;
    var cls    = chgPct >= 0 ? 'up' : 'down';
    return '<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f5f5f5">' +
      '<span style="font-weight:600">' + (idx.indexname || idx.symbol || '—') + '</span>' +
      '<div style="text-align:right">' +
        '<div class="kpi-value" style="font-size:14px">' + (close ? close.toLocaleString('en-IN', {minimumFractionDigits:2}) : '—') + '</div>' +
        '<div class="' + cls + '" style="font-size:10px">' + (chgPct >= 0 ? '+' : '') + Number(chgPct).toFixed(2) + '%</div>' +
      '</div>' +
    '</div>';
  }).join('');
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  // Start clock immediately
  const clockEl = document.getElementById('ist-time');
  if (clockEl) {
    clockEl.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false, timeZone: 'Asia/Kolkata' }) + ' IST';
  }

  // Check health (sets mktStatus)
  await checkHealth();

  // Load overview data (always loads from historical/cached backend data)
  loadOverview();
}

// Run on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
