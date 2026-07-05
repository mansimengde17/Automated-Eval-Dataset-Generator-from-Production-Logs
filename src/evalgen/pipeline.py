"""The nightly pipeline: logs in, labeled eval dataset out."""

from __future__ import annotations

from .cluster import Clusterer, flag_anomalies
from .ingest import normalize, sample
from .labeler import build_test_case, deduplicate, route
from .store import DatasetStore


def run_pipeline(raw_logs: list[dict], store: DatasetStore,
                 sample_mode: str = "signal_boosted",
                 sample_target: int = 200) -> dict:
    ingested = normalize(raw_logs)
    sampled = sample(ingested["entries"], sample_mode, sample_target)

    clustering = Clusterer().run(sampled)
    outlier_ids = {entry["id"] for entry in clustering["outliers"]}
    cluster_of = {}
    for cluster in clustering["clusters"]:
        for member in cluster["members"]:
            cluster_of[member["id"]] = cluster["label"]

    anomalies = flag_anomalies(sampled, outlier_ids)
    anomaly_ids = {entry["id"] for entry in anomalies}
    # candidates: every anomaly plus a slice of each cluster for coverage
    candidates = list(anomalies)
    for cluster in clustering["clusters"]:
        healthy = [m for m in cluster["members"]
                   if m["id"] not in anomaly_ids]
        candidates.extend(healthy[:3])

    cases = [build_test_case(entry, cluster_of.get(entry["id"], "outlier"))
             for entry in candidates]
    deduped = deduplicate(cases, store.dataset())
    routed = route(deduped["added"])
    store.add_cases(routed["auto_accepted"], status="active")
    store.add_cases(routed["review_queue"], status="pending_review")

    return {"ingested": len(ingested["entries"]),
            "dropped": ingested["dropped"],
            "pii_redactions": ingested["pii_redactions"],
            "sampled": len(sampled),
            "clusters": [{"id": c["id"], "label": c["label"],
                          "size": c["size"]}
                         for c in clustering["clusters"]],
            "outliers": len(clustering["outliers"]),
            "anomalies": len(anomalies),
            "candidates": len(cases),
            "duplicates_skipped": len(deduped["skipped"]),
            "auto_accepted": len(routed["auto_accepted"]),
            "sent_to_review": len(routed["review_queue"]),
            "dataset": store.growth_stats()}
