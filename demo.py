"""End to end demo: production logs become a growing eval dataset.

1. Two weeks of simulated production traffic is ingested and redacted.
2. Clustering reveals the natural categories and the outliers.
3. Auto labeling turns anomalies and coverage samples into test cases.
4. Low confidence labels land in the review queue, the rest go live.
5. The eval runner scores a healthy model, then a degraded model, and
   the regression detector flags exactly what broke.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

if os.path.exists("demo_evalgen.db"):
    os.remove("demo_evalgen.db")

from evalgen.logsim import generate_logs
from evalgen.pipeline import run_pipeline
from evalgen.runner import detect_regressions, run_eval
from evalgen.store import DatasetStore


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main() -> None:
    store = DatasetStore("demo_evalgen.db")

    section("Night 1: pipeline run over 600 production log entries")
    report = run_pipeline(generate_logs(600, seed=7), store)
    print(f"  ingested {report['ingested']}, dropped {report['dropped']},"
          f" PII redactions {report['pii_redactions']}")
    print(f"  clusters found: {len(report['clusters'])}")
    for cluster in report["clusters"][:5]:
        print(f"    {cluster['id']} ({cluster['size']}): {cluster['label']}")
    print(f"  outliers {report['outliers']}, anomalies {report['anomalies']}")
    print(f"  auto accepted {report['auto_accepted']},"
          f" sent to review {report['sent_to_review']},"
          f" duplicates skipped {report['duplicates_skipped']}")

    section("Night 2: new traffic, dedup keeps the dataset clean")
    report2 = run_pipeline(generate_logs(600, seed=11), store)
    print(f"  candidates {report2['candidates']},"
          f" duplicates skipped {report2['duplicates_skipped']},"
          f" auto accepted {report2['auto_accepted']}")
    stats = store.growth_stats()
    print(f"  dataset now {stats['active_total']} active cases")
    print(f"  by difficulty: {stats['by_difficulty']}")

    section("Human review: approving two queued cases")
    queue = store.review_queue()
    print(f"  review queue holds {len(queue)} low confidence cases")
    for case in queue[:2]:
        store.resolve_review(case["case_id"], "approve", "reviewer-1")
        print(f"  approved {case['case_id']}"
              f" (judge scores {case['judge_scores']})")

    section("Eval run: healthy model vs degraded model")
    dataset = store.dataset()
    healthy = run_eval("assistant-v1", dataset)
    store.save_run("assistant-v1", healthy)
    print(f"  assistant-v1 pass rate {healthy['pass_rate']}"
          f" over {healthy['total']} cases")
    degraded = run_eval("assistant-v2-degraded", dataset)
    store.save_run("assistant-v2-degraded", degraded)
    print(f"  assistant-v2-degraded pass rate {degraded['pass_rate']}")

    regression = detect_regressions(healthy, degraded)
    print(f"  verdict: {regression['verdict'].upper()},"
          f" pass rate change {regression['pass_rate_change']}")
    print(f"  new failures: {len(regression['new_failures'])}")
    for category, shift in regression["category_shifts"].items():
        print(f"    {category}: {shift['from']} -> {shift['to']}")
    print("\nDemo complete. Start the API with:"
          " uvicorn evalgen.api:app --app-dir src")


if __name__ == "__main__":
    main()
