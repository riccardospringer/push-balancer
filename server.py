#!/usr/bin/env python3
"""Story Radar standalone server — port 8090."""

import http.server
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

sys.path.insert(0, os.path.dirname(__file__))

from story_radar.api import StoryRadarHTTPAPI
from story_radar.rss_ingestor import start_background_ingestor

_api = StoryRadarHTTPAPI()
start_background_ingestor(_api.service, interval=300)

# ── Dashboard HTML (inline) ───────────────────────────────────────────────────
_DASHBOARD_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Story Radar</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #f4f5f7; --white: #fff; --border: #e5e7eb;
      --text: #111; --muted: #6b7280; --tertiary: #9ca3af;
      --accent: #4f46e5; --accent-light: #eef2ff;
      --green: #059669; --green-bg: #ecfdf5; --green-border: #a7f3d0;
      --red: #dc2626; --red-bg: #fef2f2;
      --amber: #d97706; --amber-bg: #fffbeb;
      --blue-bg: #eff6ff; --blue-border: #bfdbfe;
    }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
    .panel { background: var(--white); border: 1px solid var(--border); border-radius: 12px;
             padding: 20px 22px; margin-bottom: 14px; }
    .app-shell { max-width: 1200px; margin: 0 auto; padding: 20px 16px; }

    /* Header */
    .app-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
    .app-kicker { font-size: 11px; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 0.8px; color: var(--accent); margin-bottom: 4px; }
    h1 { font-size: 22px; font-weight: 700; }
    .app-subtitle { font-size: 13px; color: var(--muted); margin-top: 4px; max-width: 520px; }
    .header-meta { display: flex; gap: 20px; flex-shrink: 0; padding-top: 4px; }
    .meta-label { font-size: 11px; color: var(--tertiary); display: block; }
    .meta-value { font-size: 13px; font-weight: 600; }

    /* Summary cards */
    .summary-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
    .summary-card { background: var(--white); border: 1px solid var(--border); border-radius: 10px;
                    padding: 12px 16px; min-width: 110px; flex: 1; }
    .summary-card.open { border-color: var(--green-border); background: var(--green-bg); }
    .summary-card.partial { border-color: var(--blue-border); background: var(--blue-bg); }
    .summary-label { font-size: 11px; color: var(--muted); display: block; margin-bottom: 2px; }
    .summary-value { font-size: 22px; font-weight: 700; display: block; }

    /* Toolbar */
    .toolbar-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field-label { font-size: 11px; font-weight: 600; color: var(--muted);
                   text-transform: uppercase; letter-spacing: 0.5px; }
    select, input[type=search] { border: 1px solid var(--border); border-radius: 8px;
                                  padding: 6px 10px; font-size: 13px; background: var(--white);
                                  color: var(--text); outline: none; min-width: 140px; }
    select:focus, input:focus { border-color: var(--accent); }
    .toolbar-actions { display: flex; gap: 8px; align-items: center; }
    .status-text { font-size: 12px; color: var(--muted); }
    button { border: 1px solid var(--border); border-radius: 8px; padding: 6px 14px;
             font-size: 13px; background: var(--white); cursor: pointer; }
    button:hover { background: var(--bg); }
    .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .btn-primary:hover { background: #4338ca; }
    .btn-train { font-size: 12px; padding: 5px 11px; }
    #train-status { font-size: 12px; }

    /* List */
    .list-head { display: grid;
                 grid-template-columns: 72px 130px 1fr 200px 80px 60px 60px 56px;
                 gap: 8px; padding: 6px 14px; font-size: 11px; font-weight: 700;
                 text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); }
    .story-list { display: flex; flex-direction: column; gap: 1px; }
    .story-row { display: grid;
                 grid-template-columns: 72px 130px 1fr 200px 80px 60px 60px 56px;
                 gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--border);
                 align-items: start; }
    .story-row:last-child { border-bottom: none; }
    .story-row.suppressed { opacity: 0.5; }

    /* Score column */
    .score-wrap { text-align: center; }
    .score-main { font-size: 20px; font-weight: 700; }
    .score-conf { font-size: 10px; color: var(--muted); margin-top: 2px; }

    /* Badges */
    .badge { display: inline-block; font-size: 11px; font-weight: 600; border-radius: 6px;
             padding: 2px 7px; white-space: nowrap; }
    .badge-open  { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
    .badge-angle { background: var(--amber-bg); color: var(--amber); border: 1px solid #fde68a; }
    .badge-partial { background: var(--blue-bg); color: #1d4ed8; border: 1px solid var(--blue-border); }
    .badge-covered { background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
    .badge-suppressed { background: #f3f4f6; color: var(--muted); border: 1px solid var(--border); font-size: 10px; }

    /* Story column */
    .story-title { font-weight: 600; font-size: 13px; margin-bottom: 3px; }
    .story-summary { font-size: 12px; color: var(--muted); }
    .story-sources { font-size: 11px; color: var(--tertiary); margin-top: 4px; }

    /* Why relevant */
    .why-label { font-size: 10px; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.5px; color: var(--muted); margin-bottom: 2px; }
    .why-text { font-size: 12px; color: var(--text); }
    .why-angle { font-size: 11px; color: var(--accent); margin-top: 4px; font-style: italic; }

    /* Section badge */
    .section-tag { font-size: 11px; color: var(--muted); }

    /* Notice */
    .notice { background: var(--blue-bg); border: 1px solid var(--blue-border);
              border-radius: 8px; padding: 8px 12px; font-size: 12px; color: #1d4ed8;
              margin-bottom: 10px; display: none; }
    .empty-state { padding: 32px; text-align: center; color: var(--muted); font-size: 13px; }

    /* Rating buttons */
    .rate-wrap { display: flex; flex-direction: column; gap: 4px; align-items: center; padding-top: 2px; }
    .rate-btn { border: 1px solid var(--border); border-radius: 6px; background: var(--white);
                cursor: pointer; font-size: 15px; padding: 3px 7px; line-height: 1; }
    .rate-btn:hover { background: var(--bg); }
    .rate-btn.voted-up   { background: var(--green-bg); border-color: var(--green-border); }
    .rate-btn.voted-down { background: var(--red-bg);   border-color: #fecaca; }
    .rate-btn:disabled { cursor: default; opacity: 0.6; }

    /* Train button states */
    .ok  { color: var(--green) !important; border-color: var(--green) !important; }
    .err { color: var(--red)   !important; border-color: var(--red)   !important; }

    @media (max-width: 800px) {
      .list-head, .story-row { grid-template-columns: 60px 100px 1fr; }
      .story-row > *:nth-child(n+4) { display: none; }
    }
  </style>
</head>
<body>
<main class="app-shell">

  <header class="panel app-header">
    <div>
      <p class="app-kicker">Story Radar</p>
      <h1>What BILD should look at right now</h1>
      <p class="app-subtitle">Prioritised editorial view — clear BILD status, open gaps first, only the filters that actually help.</p>
    </div>
    <div class="header-meta">
      <div>
        <span class="meta-label">Last updated</span>
        <strong class="meta-value" id="last-updated">–</strong>
      </div>
      <div>
        <span class="meta-label">Stories loaded</span>
        <strong class="meta-value" id="feed-status">–</strong>
      </div>
      <div>
        <span class="meta-label">Ratings for training</span>
        <strong class="meta-value" id="rating-count" style="color:var(--accent)">0</strong>
      </div>
    </div>
  </header>

  <div class="summary-grid">
    <div class="summary-card"><span class="summary-label">Stories</span><strong class="summary-value" id="stat-total">–</strong></div>
    <div class="summary-card open"><span class="summary-label">Not on BILD</span><strong class="summary-value" id="stat-open">–</strong></div>
    <div class="summary-card partial"><span class="summary-label">Partially covered</span><strong class="summary-value" id="stat-partial">–</strong></div>
    <div class="summary-card"><span class="summary-label">Avg. score</span><strong class="summary-value" id="stat-avg">–</strong></div>
  </div>

  <section class="panel">
    <div class="toolbar-grid">
      <label class="field">
        <span class="field-label">Search</span>
        <input id="search" type="search" placeholder="Title, summary or source">
      </label>
      <label class="field">
        <span class="field-label">Section</span>
        <select id="section">
          <option value="">All sections</option>
          <option value="crime">Crime</option>
          <option value="consumer">Consumer</option>
          <option value="politics">Politics</option>
          <option value="sport">Sport</option>
          <option value="promi">Celebrity</option>
          <option value="other">Other</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">BILD status</span>
        <select id="coverage">
          <option value="focus" selected>Open gaps first</option>
          <option value="none">Not on BILD</option>
          <option value="partial">Partially on BILD</option>
          <option value="covered">Already on BILD</option>
          <option value="all">Show all</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Sort by</span>
        <select id="sort">
          <option value="score" selected>Score (desc)</option>
          <option value="urgency">Most urgent</option>
          <option value="newest">Newest first</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Show suppressed</span>
        <select id="suppressed">
          <option value="false" selected>Hidden stories off</option>
          <option value="true">Include suppressed</option>
        </select>
      </label>
    </div>
    <div class="toolbar-actions" style="justify-content:space-between">
      <div style="display:flex;gap:8px;align-items:center">
        <span class="status-text" id="status">Loading…</span>
        <button onclick="resetFilters()">Reset</button>
        <button class="btn-primary" onclick="load()">Reload</button>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <span id="train-status" style="font-size:12px"></span>
        <button class="btn-train" id="train-btn" onclick="trainModel()">Train model</button>
      </div>
    </div>
  </section>

  <div class="notice" id="notice"></div>

  <section class="panel" style="padding:0;overflow:hidden">
    <div class="list-head">
      <span>Score</span><span>BILD Status</span><span>Story</span>
      <span>Why relevant</span><span>Section</span><span>Sources</span><span>Age</span><span>Rate</span>
    </div>
    <div class="story-list" id="story-list">
      <div class="empty-state">Loading Story Radar…</div>
    </div>
  </section>

</main>
<script>
const API = '';
let _items = [];
const _ratings = {};  // cluster_id -> 'up' | 'down'

const SECTION_MAP = {
  crime: 'Crime', consumer: 'Consumer', politics: 'Politics',
  sport: 'Sport', promi: 'Celebrity', other: 'Other',
};
const COVERAGE_BADGE = {
  not_covered:      { label: 'Not on BILD',       cls: 'badge-open' },
  angle_gap:        { label: 'Angle missing',      cls: 'badge-angle' },
  partially_covered:{ label: 'Partially covered',  cls: 'badge-partial' },
  follow_up:        { label: 'Update recommended', cls: 'badge-angle' },
  already_covered:  { label: 'Already covered',    cls: 'badge-covered' },
};
const SUPPRESS_LABEL = {
  already_covered: 'already covered', standard_noise: 'too much noise',
  low_confidence: 'low confidence',   weak_story: 'weak story',
};

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt(v) { return Number.isFinite(Number(v)) ? Number(v).toFixed(2) : '–'; }
function relTime(iso) {
  if (!iso) return '–';
  const diff = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (diff < 60)  return diff + 'm';
  if (diff < 1440) return Math.round(diff/60) + 'h';
  return Math.round(diff/1440) + 'd';
}
function sectionKey(topics) {
  const t = (topics||[]).map(s=>s.toLowerCase());
  if (t.some(x=>['crime','public safety','police','justice'].includes(x))) return 'crime';
  if (t.some(x=>['consumer','service','transport','money','health'].includes(x))) return 'consumer';
  if (t.some(x=>['politics','policy','election','energy'].includes(x))) return 'politics';
  if (t.some(x=>['sport','football','soccer','bundesliga'].includes(x))) return 'sport';
  if (t.some(x=>['promi','celebrity','royal','entertainment'].includes(x))) return 'promi';
  return 'other';
}

function renderRow(item) {
  const c = item.cluster || {}, gap = item.gap_assessment || {};
  const ml = item.ml_score || {}, r = item.ranking || {};
  const expl = item.explanations || {};
  const cov = COVERAGE_BADGE[gap.coverage_status] || { label: 'Unknown', cls: 'badge-covered' };
  const secKey = sectionKey(c.topics);
  const secLabel = SECTION_MAP[secKey] || 'Other';
  const conf = Number(r.confidence||0);
  const confLabel = conf>=0.8?'high':conf>=0.6?'medium':'low';
  const suppBadge = r.suppressed
    ? `<div style="margin-top:4px"><span class="badge badge-suppressed">hidden: ${esc(SUPPRESS_LABEL[r.suppression_reason]||'suppressed')}</span></div>`
    : '';
  const whyNow = expl.why_now || '';
  const angle  = expl.recommended_angle || '';
  return `<div class="story-row${r.suppressed?' suppressed':''}">
    <div class="score-wrap">
      <div class="score-main">${esc(fmt(r.final_score))}</div>
      <div class="score-conf">Conf: ${esc(confLabel)}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:2px">ML ${esc(fmt(ml.relevance_score))}</div>
    </div>
    <div>
      <span class="badge ${esc(cov.cls)}">${esc(cov.label)}</span>
      ${suppBadge}
    </div>
    <div>
      <div class="story-title">${esc(c.title||'No title')}</div>
      <div class="story-summary">${esc((c.summary||'').slice(0,120))}</div>
      <div class="story-sources">${esc(String(c.source_count||0))} sources · ${esc(String(c.document_count||0))} docs</div>
    </div>
    <div>
      <div class="why-label">Why relevant</div>
      <div class="why-text">${esc(expl.why_relevant||r.ranking_reason||'–')}</div>
      ${whyNow ? `<div class="why-label" style="margin-top:6px">Why now</div><div class="why-text">${esc(whyNow)}</div>` : ''}
      ${angle  ? `<div class="why-angle">${esc(angle)}</div>` : ''}
    </div>
    <div class="section-tag">${esc(secLabel)}</div>
    <div style="font-size:12px;color:var(--muted)">${esc(String(c.source_count||0))}</div>
    <div style="font-size:12px;color:var(--muted)">${esc(relTime(c.last_seen_at||c.first_seen_at))}</div>
    <div class="rate-wrap">
      <button class="rate-btn${_ratings[cid]==='up'?' voted-up':''}"
              id="rb-up-${esc(cid)}"
              onclick="rate('${esc(cid)}','up')"
              title="Would cover (picked_up)">👍</button>
      <button class="rate-btn${_ratings[cid]==='down'?' voted-down':''}"
              id="rb-down-${esc(cid)}"
              onclick="rate('${esc(cid)}','down')"
              title="Skip (dismissed)">👎</button>
    </div>
  </div>`;
}

function applyFilters() {
  const q      = document.getElementById('search').value.toLowerCase();
  const sec    = document.getElementById('section').value;
  const cov    = document.getElementById('coverage').value;
  const sortBy = document.getElementById('sort').value;
  const inclSupp = document.getElementById('suppressed').value === 'true';

  let items = _items.filter(item => {
    const c = item.cluster || {}, gap = item.gap_assessment || {}, r = item.ranking || {};
    if (!inclSupp && r.suppressed) return false;
    if (q) {
      const hay = ((c.title||'')+(c.summary||'')).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (sec) {
      const key = sectionKey(c.topics);
      if (key !== sec) return false;
    }
    if (cov !== 'all') {
      const st = gap.coverage_status || '';
      if (cov === 'focus') {
        if (!['not_covered','angle_gap','follow_up'].includes(st)) return false;
      } else if (cov === 'none') {
        if (st !== 'not_covered') return false;
      } else if (cov === 'partial') {
        if (!['partially_covered','angle_gap','follow_up'].includes(st)) return false;
      } else if (cov === 'covered') {
        if (st !== 'already_covered') return false;
      }
    }
    return true;
  });

  if (sortBy === 'urgency') {
    items.sort((a,b) => Number(((b.features||{}).urgency_score)||0) - Number(((a.features||{}).urgency_score)||0));
  } else if (sortBy === 'newest') {
    items.sort((a,b) => new Date(b.cluster?.last_seen_at||0) - new Date(a.cluster?.last_seen_at||0));
  }

  // Update summary
  const all = _items.filter(i=>!((i.ranking||{}).suppressed));
  const open = all.filter(i=>(i.gap_assessment||{}).coverage_status==='not_covered').length;
  const partial = all.filter(i=>['partially_covered','angle_gap','follow_up'].includes((i.gap_assessment||{}).coverage_status||'')).length;
  const avg = all.length ? all.reduce((s,i)=>s+Number(((i.ranking||{}).final_score)||0),0)/all.length : 0;
  document.getElementById('stat-total').textContent   = all.length;
  document.getElementById('stat-open').textContent    = open;
  document.getElementById('stat-partial').textContent = partial;
  document.getElementById('stat-avg').textContent     = avg.toFixed(2);
  document.getElementById('feed-status').textContent  = items.length + ' visible';

  const list = document.getElementById('story-list');
  if (!items.length) {
    list.innerHTML = '<div class="empty-state">No stories match the current filters.</div>';
    return;
  }
  list.innerHTML = items.map(renderRow).join('');
  document.getElementById('rating-count').textContent = Object.keys(_ratings).length;
}

async function load(silent=false) {
  const suppressed = document.getElementById('suppressed').value === 'true';
  if (!silent) document.getElementById('status').textContent = 'Loading…';
  try {
    const r = await fetch(API + '/api/story-radar/ranked?model_variant=ml&include_suppressed=' + suppressed);
    const data = await r.json();
    _items = Array.isArray(data.items) ? data.items : [];
    const ts = data.generated_at ? new Date(data.generated_at).toLocaleTimeString() : '–';
    document.getElementById('last-updated').textContent = ts;
    document.getElementById('status').textContent = 'Ready';
    const n = document.getElementById('notice');
    if (data.generated_at) { n.textContent = 'Updated ' + ts; n.style.display = 'block'; }
    applyFilters();
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
    document.getElementById('story-list').innerHTML =
      '<div class="empty-state">Load failed: ' + esc(e.message) + ' <button onclick="load()">Retry</button></div>';
  }
}

async function rate(clusterId, dir) {
  const action = dir === 'up' ? 'picked_up' : 'dismissed';
  const prev = _ratings[clusterId];
  if (prev === dir) return;  // already rated this way

  _ratings[clusterId] = dir;

  // Update button visuals immediately
  const upBtn   = document.getElementById('rb-up-'   + clusterId);
  const downBtn = document.getElementById('rb-down-' + clusterId);
  if (upBtn)   { upBtn.className   = 'rate-btn' + (dir==='up'   ? ' voted-up'   : ''); }
  if (downBtn) { downBtn.className = 'rate-btn' + (dir==='down' ? ' voted-down' : ''); }

  // Update counter
  document.getElementById('rating-count').textContent = Object.keys(_ratings).length;

  try {
    await fetch(API + '/api/story-radar/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ cluster_id: clusterId, editor_id: 'editor', action }),
    });
  } catch(e) {
    console.warn('Rating failed:', e.message);
  }
}

async function trainModel() {
  const btn = document.getElementById('train-btn');
  const st  = document.getElementById('train-status');
  if (btn.disabled) return;
  btn.disabled = true; btn.textContent = 'Training…'; btn.className = 'btn-train';
  st.textContent = '';
  try {
    const r = await fetch(API + '/api/story-radar/train', { method: 'POST' });
    const d = await r.json();
    if (d.ok) {
      btn.textContent = '✓ Trained'; btn.className = 'btn-train ok';
      st.textContent = d.trained_rows + ' rows · ' + d.scorer; st.style.color = 'var(--green)';
      load();
    } else {
      btn.textContent = '✗ Failed'; btn.className = 'btn-train err';
      st.textContent = d.error || 'Training failed'; st.style.color = 'var(--red)';
    }
  } catch(e) {
    btn.textContent = '✗ Error'; btn.className = 'btn-train err';
    st.textContent = e.message; st.style.color = 'var(--red)';
  } finally {
    btn.disabled = false;
    setTimeout(() => {
      btn.textContent = 'Train model'; btn.className = 'btn-train';
      st.textContent = ''; st.style.color = '';
    }, 6000);
  }
}

function resetFilters() {
  document.getElementById('search').value = '';
  document.getElementById('section').value = '';
  document.getElementById('coverage').value = 'focus';
  document.getElementById('sort').value = 'score';
  document.getElementById('suppressed').value = 'false';
  applyFilters();
}

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('section').addEventListener('change', applyFilters);
document.getElementById('coverage').addEventListener('change', applyFilters);
document.getElementById('sort').addEventListener('change', applyFilters);
document.getElementById('suppressed').addEventListener('change', () => load());

load();
setInterval(() => load(true), 60000);
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json_response(self, data, status=200, ensure_ascii=True):
        body = json.dumps(data, ensure_ascii=ensure_ascii).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, msg):
        self._json_response({"error": msg}, status=status)

    def _html_response(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/dashboard", "/dashboard/"):
            self._html_response(_DASHBOARD_HTML)
            return
        if _api.handle_get(self):
            return
        self._error(404, "Not found")

    def do_POST(self):
        if _api.handle_post(self):
            return
        self._error(404, "Not found")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"Story Radar  →  http://127.0.0.1:{port}/dashboard/")
    server.serve_forever()
