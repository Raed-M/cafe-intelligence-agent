const reportStyles = `
<style>
  :root{--ink:#17212b;--muted:#5e6872;--paper:#fbfaf7;--line:#d9d5cc;--copper:#a85f2e;--palm:#2f6b55;--night:#0b2036}
  *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.72}
  body>header,body>main,body>.report-route,body>h1,body>h2,body>h3,body>p,body>ul,body>ol,body>table,body>section,body>details{width:min(100% - 40px,900px);margin-inline:auto}
  body{padding:42px 0 72px}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;color:var(--night);line-height:1.18;text-wrap:balance}h1{margin-block:0 30px;font-size:clamp(2.15rem,6vw,4.2rem);letter-spacing:-.035em}h2{margin-block:44px 18px;border-block-start:1px solid var(--line);padding-block-start:24px;font-size:clamp(1.55rem,3vw,2.25rem)}h3{margin-block:25px 10px;font-size:1.3rem}
  p,li{max-width:76ch}strong{color:#101820}a{color:var(--palm)}code{border-radius:5px;background:#edf0ed;padding:2px 5px;font-size:.88em;overflow-wrap:anywhere}
  .report-route{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:12px;margin-block:4px 36px;border:1px solid #c9d5cf;border-radius:18px;background:#f1f6f3;padding:18px 22px;color:var(--night)}.report-route div{display:grid;justify-items:center;gap:7px;text-align:center}.report-route svg{width:28px;height:28px;color:var(--palm)}.report-route b{font-size:.82rem}.report-route span{color:#8a6a52;font-size:1.2rem}.report-route-note{width:min(100% - 40px,900px);margin:-24px auto 38px;color:var(--muted);font-size:.84rem;text-align:center}
  table{display:table;margin-block:20px;border:1px solid var(--line);border-radius:12px;border-collapse:separate;border-spacing:0;overflow:hidden;font-size:.92rem}th{background:var(--night);color:white;font-weight:700;text-align:start}th,td{border-block-end:1px solid var(--line);padding:11px 13px;vertical-align:top}tr:last-child td{border-block-end:0}tbody tr:nth-child(even){background:#f3f1eb}
  blockquote{width:min(100% - 40px,820px);margin:24px auto;border-inline-start:4px solid var(--copper);background:#f5eee7;padding:16px 20px;color:#423329}
  .technical-appendix{margin-block:36px;border:1px solid var(--line);border-radius:14px;background:#f2f1ed}.technical-appendix summary{cursor:pointer;padding:16px 18px;color:var(--night);font-weight:750}.technical-appendix summary small{display:block;margin-top:3px;color:var(--muted);font-weight:400}.technical-appendix>div{border-block-start:1px solid var(--line);padding:0 18px 20px}.technical-appendix h2{width:100%;margin-top:18px}
  img,svg{max-width:100%}pre{max-width:100%;overflow:auto;border-radius:10px;background:#111c27;color:#eef3f5;padding:16px;white-space:pre-wrap}.content-idea{border:1px solid var(--line);border-radius:12px;padding:16px}
  @media(max-width:700px){body{padding-top:26px}body>header,body>main,body>.report-route,body>h1,body>h2,body>h3,body>p,body>ul,body>ol,body>table,body>section,body>details{width:min(100% - 24px,900px)}.report-route{grid-template-columns:1fr;gap:8px}.report-route span{transform:rotate(90deg)}table{display:block;overflow-x:auto;white-space:nowrap}}
  @media print{body{padding:0;background:white}.report-route{break-inside:avoid}.technical-appendix{display:none}}
</style>`;

const readingRoute = `
<div class="report-route" role="img" aria-label="Report reading path: source data, verified findings, human decision">
  <div><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg><b>Source data</b></div><span aria-hidden="true">→</span>
  <div><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m9 12 2 2 4-5"/><circle cx="12" cy="12" r="9"/></svg><b>Verified findings</b></div><span aria-hidden="true">→</span>
  <div><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 11h10M7 15h7M8 3h8l4 4v14H4V3h4Z"/></svg><b>Human decision</b></div>
</div><p class="report-route-note">Start with the executive summary. Use the evidence sections for detail; external source dumps stay folded in the technical appendix.</p>`;

export function enhanceReportHtml(html: string) {
  let output = html.includes("</head>") ? html.replace("</head>", `${reportStyles}</head>`) : `${reportStyles}${html}`;
  
  const context = /(<section>\s*<h2[^>]*>\s*Local,\s*Calendar[^<]*<\/h2>[\s\S]*?<\/section>)/i;
  output = output.replace(context, `<details class="technical-appendix"><summary>Local context and external sources<small>Open only when you need the raw prayer, weather, or event references.</small></summary><div>$1</div></details>`);
  
  return output.replace(/<body([^>]*)>/i, `<body$1>${readingRoute}`);
}
