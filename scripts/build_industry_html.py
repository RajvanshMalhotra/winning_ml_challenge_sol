"""Render artifacts/v2/industry_clusters_<model>.json into a self-contained HTML report.
usage: python scripts/build_industry_html.py <json_path> <out_html>"""
import html
import json
import sys

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
clusters = d["clusters"]
countries = sorted({c for cl in clusters for c in cl["n"]}, key=lambda c: ["US", "India", "France"].index(c)
                   if c in ("US", "India", "France") else 9)
rows = "".join(
    f"<tr><td>{html.escape(m['s1'])}</td><td>{html.escape(m['s1_cluster'])}</td>"
    f"<td>{html.escape(m['cand'])}</td><td>{html.escape(m['cand_cluster'])}</td></tr>"
    for m in d["mismatch_examples"])
by_c = " · ".join(f"{c}: {v:.1%}" for c, v in d["same_rate_by_country"].items())

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industry Clusters</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {{ color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#8a8984; --grid:#e6e5e1; --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --dim:#d4d3ce; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --surface-1:#1a1a19;
  --surface-2:#242422; --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#8f8e86; --grid:#33332f;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --dim:#44443f; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface-1:#1a1a19; --surface-2:#242422; --text-primary:#fff;
  --text-secondary:#c3c2b7; --muted:#8f8e86; --grid:#33332f; --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --dim:#44443f; }}
body {{ margin:0; background:var(--surface-1); color:var(--text-primary); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:36px 0 4px; }}
p.sub {{ color:var(--text-secondary); margin:0 0 12px; }}
.hero {{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0; }}
.tile {{ background:var(--surface-2); border-radius:10px; padding:14px 18px; min-width:200px; flex:1; }}
.tile .v {{ font-size:34px; font-weight:650; }} .tile .l {{ color:var(--text-secondary); font-size:13px; }}
.chart {{ width:100%; }}
select {{ font:inherit; padding:4px 8px; background:var(--surface-2); color:var(--text-primary); border:1px solid var(--grid); border-radius:6px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} td,th {{ border-bottom:1px solid var(--grid); padding:6px 8px; text-align:left; }}
th {{ color:var(--text-secondary); font-weight:600; }} .scroll {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Industry clusters of business names</h1>
<p class="sub">{d['k']} clusters from <b>{html.escape(d['model'])}</b> embeddings of the business name only (k-means, cosine).
Hypothesis: records of the same business land in the same industry cluster.</p>

<div class="hero">
 <div class="tile"><div class="v">{d['same_rate']:.1%}</div><div class="l">of {d['n_pairs']:,} true match pairs share a cluster</div></div>
 <div class="tile"><div class="v">{d['chance']:.1%}</div><div class="l">expected by chance (random cluster assignment)</div></div>
 <div class="tile"><div class="v">{1 - d['same_rate']:.1%}</div><div class="l">of true matches would be <b>lost</b> if we only searched within a cluster</div></div>
</div>
<p class="sub">By country: {by_c}</p>

<h2>Cluster sizes</h2><p class="sub">Businesses per cluster (train S1 sample + France test sample). Hover for example names.</p>
<div id="sizes" class="chart"></div>
<h2>Same-cluster rate per cluster</h2><p class="sub">Share of that cluster's true match pairs whose matched record lands in the same cluster. Dashed line = overall.</p>
<div id="rates" class="chart"></div>
<h2>Country mix per cluster</h2><p class="sub">Do French businesses fall into the same industries as US / Indian ones?</p>
<div id="mix" class="chart"></div>
<h2>Map of names</h2><p class="sub">2-D UMAP of {len(d['points']):,} names. Highlight one cluster:
 <select id="pick"></select></p>
<div id="map" class="chart" style="height:560px"></div>
<h2>Examples of true matches in different clusters</h2>
<div class="scroll"><table><tr><th>S1 name</th><th>S1 cluster</th><th>Matched record</th><th>Its cluster</th></tr>{rows}</table></div>
</main>
<script>
const D = {json.dumps({"clusters": clusters, "points": d["points"], "overall": d["same_rate"], "countries": countries})};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const lbl = c => `${{c.id}} · ${{c.label || "(mixed)"}}`;
function base(h) {{ return {{ height:h, margin:{{l:260,r:24,t:8,b:36}}, paper_bgcolor:css('--surface-1'), plot_bgcolor:css('--surface-1'),
  font:{{color:css('--text-primary'), size:12}}, xaxis:{{gridcolor:css('--grid'), zeroline:false}}, yaxis:{{automargin:true}},
  hoverlabel:{{bgcolor:css('--surface-2'), font:{{color:css('--text-primary')}}}}, bargap:0.25 }}; }}
const cfg = {{displayModeBar:false, responsive:true}};
function draw() {{
  const C = D.clusters, tot = c => Object.values(c.n).reduce((a,b)=>a+b,0);
  const bySize = [...C].sort((a,b)=>tot(a)-tot(b));
  Plotly.react('sizes', [{{type:'bar', orientation:'h', y:bySize.map(lbl), x:bySize.map(tot), marker:{{color:css('--series-1')}},
    customdata:bySize.map(c=>c.examples.join('<br>')), hovertemplate:'<b>%{{y}}</b><br>%{{x:,}} businesses<br><br>%{{customdata}}<extra></extra>'}}],
    base(Math.max(420, C.length*22)), cfg);
  const R = C.filter(c=>c.same_rate!==null).sort((a,b)=>a.same_rate-b.same_rate);
  const lr = base(Math.max(420, R.length*22)); lr.xaxis.tickformat='.0%'; lr.xaxis.range=[0,1];
  lr.shapes=[{{type:'line', x0:D.overall, x1:D.overall, yref:'paper', y0:0, y1:1, line:{{color:css('--text-secondary'), dash:'dash', width:1.5}}}}];
  Plotly.react('rates', [{{type:'bar', orientation:'h', y:R.map(lbl), x:R.map(c=>c.same_rate), marker:{{color:css('--series-1')}},
    customdata:R.map(c=>c.n_pairs), hovertemplate:'<b>%{{y}}</b><br>%{{x:.1%}} of %{{customdata:,}} pairs stay in-cluster<extra></extra>'}}], lr, cfg);
  const cols = [css('--series-1'), css('--series-2'), css('--series-3')];
  const lm = base(Math.max(420, C.length*22)); lm.barmode='stack'; lm.xaxis.tickformat='.0%'; lm.xaxis.range=[0,1];
  lm.legend={{orientation:'h', y:1.04, x:0}}; lm.margin.t=36;
  Plotly.react('mix', D.countries.map((ct,i)=>({{type:'bar', orientation:'h', name:ct, y:bySize.map(lbl),
    x:bySize.map(c=>(c.n[ct]||0)/tot(c)), marker:{{color:cols[i%3], line:{{color:css('--surface-1'), width:2}}}},
    hovertemplate:`<b>%{{y}}</b><br>${{ct}}: %{{x:.1%}}<extra></extra>`}})), lm, cfg);
  drawMap();
}}
function drawMap() {{
  const k = +document.getElementById('pick').value, P = D.points;
  const on = P.filter(p=>p.c===k), off = P.filter(p=>p.c!==k);
  const tr = (pts, color, size, name) => ({{type:'scattergl', mode:'markers', name, x:pts.map(p=>p.x), y:pts.map(p=>p.y),
    text:pts.map(p=>`${{p.name}} (${{p.country}})`), hovertemplate:'%{{text}}<extra></extra>', marker:{{color, size, line:{{width:0}}}}}});
  const l = base(560); l.margin={{l:8,r:8,t:8,b:8}}; l.xaxis={{visible:false}}; l.yaxis={{visible:false}}; l.showlegend=false;
  Plotly.react('map', [tr(off, css('--dim'), 4, 'other'), tr(on, css('--series-1'), 8, 'selected')], l, cfg);
}}
const sel = document.getElementById('pick');
[...D.clusters].sort((a,b)=>a.id-b.id).forEach(c=>{{ const o=document.createElement('option'); o.value=c.id; o.textContent=lbl(c); sel.append(o); }});
sel.onchange = drawMap;
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script></body></html>"""
open(out, "w").write(page)
print("wrote", out)
