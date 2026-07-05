"""Semantic clustering and outlier detection over sampled prompts.

Embeddings are bag of words vectors so the pipeline runs offline; the
clustering is a greedy density pass in the spirit of HDBSCAN: points
join a cluster when close enough to its centroid, everything else is
noise, and noise points are exactly the eval candidates we care about.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {"the", "a", "an", "is", "are", "how", "do", "i", "my", "to",
             "for", "of", "and", "can", "what", "why", "was", "did"}


def embed(text: str) -> dict[str, float]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower())
              if t not in STOPWORDS]
    counts = Counter(tokens)
    counts.update({f"{a}_{b}": 1 for a, b in zip(tokens, tokens[1:])})
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {token: value / norm for token, value in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(token, 0.0) for token, value in a.items())


class Clusterer:
    def __init__(self, join_threshold: float = 0.35, min_size: int = 4):
        self.join_threshold = join_threshold
        self.min_size = min_size

    def run(self, entries: list[dict]) -> dict:
        vectors = [(entry, embed(entry["prompt"])) for entry in entries]
        clusters: list[dict] = []
        for entry, vector in vectors:
            best, best_sim = None, 0.0
            for cluster in clusters:
                sim = cosine(vector, cluster["centroid"])
                if sim > best_sim:
                    best, best_sim = cluster, sim
            if best is not None and best_sim >= self.join_threshold:
                best["members"].append(entry)
                # running centroid update
                n = len(best["members"])
                centroid = best["centroid"]
                for token, value in vector.items():
                    centroid[token] = centroid.get(token, 0.0) \
                        + (value - centroid.get(token, 0.0)) / n
            else:
                clusters.append({"centroid": dict(vector),
                                 "members": [entry]})
        real, noise = [], []
        for cluster in clusters:
            if len(cluster["members"]) >= self.min_size:
                real.append(cluster)
            else:
                noise.extend(cluster["members"])
        for index, cluster in enumerate(
                sorted(real, key=lambda c: -len(c["members"]))):
            top_terms = sorted(cluster["centroid"].items(),
                               key=lambda kv: -kv[1])
            cluster["id"] = f"cluster-{index:02d}"
            cluster["label"] = " / ".join(
                t for t, _ in top_terms if "_" not in t)[:60]
            cluster["size"] = len(cluster["members"])
        return {"clusters": real, "outliers": noise}


def flag_anomalies(entries: list[dict], outlier_ids: set[str]) -> list[dict]:
    """Combine structural outliers with behavioral anomaly signals."""
    flagged = []
    for entry in entries:
        reasons = []
        if entry["id"] in outlier_ids:
            reasons.append("did not cluster (novel request)")
        if entry.get("retried"):
            reasons.append("user retried after this response")
        if entry.get("feedback") == "thumbs_down":
            reasons.append("negative user feedback")
        if entry["output_tokens"] < 35:
            reasons.append("unusually short response")
        if "ignore previous instructions" in entry["prompt"].lower():
            reasons.append("prompt injection attempt")
        if reasons:
            flagged.append({**entry, "anomaly_reasons": reasons})
    return flagged
