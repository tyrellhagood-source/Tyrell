"""Builds the final rep report: a JSON blob (for programmatic use) and a
self-contained HTML "report card" (grade, pros/cons, drills, situation,
and a link to the annotated highlight clip) meant to be opened locally
after running the pipeline.
"""
import json
from dataclasses import asdict

from grading import GradeResult
from metrics import CoverageMetrics, RouteMetrics
from schema import PlayerIdentity, Situation


def build_report_dict(situation: Situation, wr_identity: PlayerIdentity, cb_identity: PlayerIdentity,
                       route: RouteMetrics, coverage: CoverageMetrics,
                       wr_grade: GradeResult, cb_grade: GradeResult, video_file: str) -> dict:
    return {
        "video_file": video_file,
        "situation": asdict(situation),
        "players": {
            "wr": asdict(wr_identity),
            "cb": asdict(cb_identity),
        },
        "route": asdict(route),
        "coverage": asdict(coverage),
        "grades": {
            "wr": asdict(wr_grade),
            "cb": asdict(cb_grade),
        },
    }


def _grade_card_html(grade: GradeResult, player_name: str) -> str:
    breakdown_rows = "".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>"
        for k, v in grade.breakdown.items()
    )
    pros = "".join(f"<li>{p}</li>" for p in grade.pros)
    cons = "".join(f"<li>{c}</li>" for c in grade.cons)
    drills = "".join(
        f"<div class='drill'><strong>{d['name']}</strong><br>"
        f"<em>{d['focus']}</em><p>{d['description']}</p></div>"
        for d in grade.drills
    ) or "<p class='muted'>No specific drills triggered -- solid rep.</p>"

    return f"""
    <div class="card">
      <h2>{grade.role} &mdash; {player_name}</h2>
      <div class="grade-badge grade-{grade.letter}">{grade.letter}</div>
      <div class="score">{grade.score}/100</div>
      <table class="breakdown">{breakdown_rows}</table>
      <h3>Pros</h3><ul class="pros">{pros}</ul>
      <h3>Cons</h3><ul class="cons">{cons}</ul>
      <h3>Recommended Drills</h3>
      {drills}
    </div>
    """


def build_html_report(report: dict, wr_grade: GradeResult, cb_grade: GradeResult, video_rel_path: str) -> str:
    sit = report["situation"]
    situation_line = " &middot; ".join(
        filter(None, [
            f"Down {sit['down']} & {sit['distance']}" if sit.get("down") else None,
            sit.get("hash_mark") and f"{sit['hash_mark']} hash",
            sit.get("offense_formation"),
            sit.get("coverage_call") and f"Coverage call: {sit['coverage_call']}",
            sit.get("level"),
        ])
    )

    route = report["route"]
    coverage = report["coverage"]

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>1-on-1 Rep Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1216; color:#eaeaea; margin:0; padding:24px; }}
  h1 {{ font-size:1.4rem; }}
  .situation {{ color:#9aa4af; margin-bottom:20px; }}
  video {{ width:100%; max-width:480px; border-radius:8px; display:block; margin-bottom:24px; }}
  .cards {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .card {{ background:#181c22; border-radius:10px; padding:18px 20px; flex:1; min-width:280px; position:relative; }}
  .grade-badge {{ position:absolute; top:16px; right:20px; font-size:2rem; font-weight:700; border-radius:50%;
                  width:52px; height:52px; display:flex; align-items:center; justify-content:center; }}
  .grade-A {{ background:#1f8a3d; }} .grade-B {{ background:#2f7ad1; }}
  .grade-C {{ background:#c99a1f; }} .grade-D {{ background:#c56a1f; }} .grade-F {{ background:#b93a3a; }}
  .score {{ color:#9aa4af; margin-bottom:10px; }}
  table.breakdown {{ width:100%; border-collapse:collapse; margin:10px 0; font-size:0.9rem; }}
  table.breakdown td {{ padding:4px 0; border-bottom:1px solid #2a2f36; }}
  ul.pros li {{ color:#7fd88f; }} ul.cons li {{ color:#e08a8a; }}
  .drill {{ background:#12151a; border-radius:8px; padding:10px 12px; margin-top:8px; font-size:0.9rem; }}
  .drill em {{ color:#9aa4af; }}
  .muted {{ color:#9aa4af; }}
  .metrics {{ background:#181c22; border-radius:10px; padding:14px 18px; margin-bottom:20px; font-size:0.92rem; }}
  .metrics b {{ color:#fff; }}
</style></head>
<body>
  <h1>1-on-1 Rep Report</h1>
  <div class="situation">{situation_line}</div>
  <video src="{video_rel_path}" controls></video>
  <div class="metrics">
    <b>Route:</b> {route['route_guess']} (confidence: {route['confidence']})<br>
    <b>Coverage:</b> {coverage['technique_guess']} / {coverage['leverage_guess']}
  </div>
  <div class="cards">
    {_grade_card_html(wr_grade, report['players']['wr']['name'])}
    {_grade_card_html(cb_grade, report['players']['cb']['name'])}
  </div>
</body></html>"""


def save_report(report: dict, wr_grade: GradeResult, cb_grade: GradeResult,
                 video_rel_path: str, json_path: str, html_path: str):
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(html_path, "w") as f:
        f.write(build_html_report(report, wr_grade, cb_grade, video_rel_path))
