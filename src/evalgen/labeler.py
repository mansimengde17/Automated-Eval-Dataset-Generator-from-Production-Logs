"""Auto labeling: quality judgment, difficulty, golden answers, routing.

Every candidate interaction becomes a labeled test case. Labels come from
multiple judge runs; when the runs agree the case is added automatically,
when they disagree it lands in the human review queue. A near duplicate
check against the existing dataset keeps the suite from bloating.
"""

from __future__ import annotations

import hashlib

from .cluster import cosine, embed

DUP_THRESHOLD = 0.92


def _seed(text: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + text).encode()).hexdigest()[:8], 16)


def judge_quality(prompt: str, response: str, run: int = 0) -> int:
    """LLM as judge stand in, deterministic per (interaction, run).

    Vague responses score low on every run; borderline interactions get
    slightly different scores per run, which is what produces genuine
    disagreement for the confidence router to handle.
    """
    if "vague non answer" in response:
        return 1 + _seed(response, str(run)) % 2       # 1..2
    base = 4 + _seed(prompt + response) % 2            # 4..5
    if _seed(prompt, "borderline") % 10 < 2:           # 20% borderline cases
        return max(1, base - _seed(response, f"jitter{run}") % 3)
    return base


def estimate_difficulty(prompt: str) -> str:
    lowered = prompt.lower()
    if "ignore previous instructions" in lowered:
        return "adversarial"
    words = len(prompt.split())
    clauses = prompt.count(",") + prompt.count(" and ")
    if "compare" in lowered or clauses >= 2 or words > 25:
        return "hard"
    if words > 12 or "(context:" in lowered:
        return "moderate"
    return "simple"


def expected_behavior(prompt: str, quality: int) -> str:
    lowered = prompt.lower()
    if "ignore previous instructions" in lowered:
        return "should_refuse"
    if not any(c.isalpha() for c in prompt.split()[0]) or \
            len(set(prompt.split())) <= 3:
        return "should_ask_clarification"
    return "should_answer"


def golden_reference(prompt: str, category: str) -> dict:
    """Reference answer for objective cases, rubric for subjective ones."""
    if category in ("billing", "account", "technical"):
        return {"type": "reference",
                "reference": f"grounded answer covering: {prompt[:60]}",
                "must_contain": [w for w in prompt.lower().split()
                                 if len(w) > 5][:3],
                "must_not_contain": ["as an ai", "i cannot help"]}
    return {"type": "rubric",
            "rubric": ["addresses the question directly",
                       "cites the relevant policy section",
                       "does not invent policy details"],
            "must_not_contain": ["guarantee", "always", "never"]}


def build_test_case(entry: dict, cluster_label: str,
                    judge_runs: int = 3) -> dict:
    scores = [judge_quality(entry["prompt"], entry["response"], run)
              for run in range(judge_runs)]
    spread = max(scores) - min(scores)
    confidence = "high" if spread <= 1 else "low"
    mean_score = round(sum(scores) / len(scores), 2)
    polarity = "positive" if mean_score >= 3.5 else "negative"
    category = entry.get("_true_category", cluster_label or "uncategorized")
    return {
        "case_id": f"case-{entry['content_hash'][:10]}",
        "source_log": entry["id"],
        "prompt": entry["prompt"],
        "category": category,
        "cluster": cluster_label,
        "difficulty": estimate_difficulty(entry["prompt"]),
        "expected_behavior": expected_behavior(entry["prompt"], mean_score),
        "expected_quality": mean_score,
        "polarity": polarity,
        "golden": golden_reference(entry["prompt"], category),
        "judge_scores": scores,
        "label_confidence": confidence,
        "anomaly_reasons": entry.get("anomaly_reasons", []),
    }


def deduplicate(candidates: list[dict], existing: list[dict]) -> dict:
    existing_vectors = [(case, embed(case["prompt"])) for case in existing]
    added, skipped = [], []
    for case in candidates:
        vector = embed(case["prompt"])
        duplicate = None
        for other, other_vector in existing_vectors:
            if cosine(vector, other_vector) > DUP_THRESHOLD:
                duplicate = other["case_id"]
                break
        if duplicate:
            skipped.append({"case_id": case["case_id"],
                            "duplicate_of": duplicate})
        else:
            added.append(case)
            existing_vectors.append((case, vector))
    return {"added": added, "skipped": skipped}


def route(cases: list[dict]) -> dict:
    auto = [c for c in cases if c["label_confidence"] == "high"]
    review = [c for c in cases if c["label_confidence"] == "low"]
    return {"auto_accepted": auto, "review_queue": review}
