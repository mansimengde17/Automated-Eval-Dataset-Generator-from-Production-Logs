"""Log ingestion: normalization, PII redaction, dedup, and sampling."""

from __future__ import annotations

import hashlib
import random
import re

REDACTION_RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"\b\+?\d{10,12}\b"), "[PHONE]"),
]

REQUIRED_FIELDS = ("id", "prompt", "response", "model", "timestamp")


def redact(text: str) -> tuple[str, int]:
    hits = 0
    for pattern, replacement in REDACTION_RULES:
        text, n = pattern.subn(replacement, text)
        hits += n
    return text, hits


def normalize(raw_logs: list[dict]) -> dict:
    """Validate schema, redact PII, and drop exact duplicates."""
    seen_hashes: set[str] = set()
    clean, dropped, redactions = [], 0, 0
    for entry in raw_logs:
        if any(field not in entry for field in REQUIRED_FIELDS):
            dropped += 1
            continue
        prompt, hits_p = redact(entry["prompt"])
        response, hits_r = redact(entry["response"])
        redactions += hits_p + hits_r
        digest = hashlib.sha256(
            (prompt + response).encode()).hexdigest()[:16]
        if digest in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(digest)
        clean.append({**entry, "prompt": prompt, "response": response,
                      "content_hash": digest})
    return {"entries": clean, "dropped": dropped,
            "pii_redactions": redactions}


def sample(entries: list[dict], mode: str = "signal_boosted",
           target: int = 200, seed: int = 3) -> list[dict]:
    """Three sampling strategies over the normalized log stream."""
    rng = random.Random(seed)
    if len(entries) <= target:
        return list(entries)
    if mode == "uniform":
        return rng.sample(entries, target)
    if mode == "stratified":
        by_key: dict[str, list[dict]] = {}
        for entry in entries:
            key = f"{entry['feature']}:{entry['model']}"
            by_key.setdefault(key, []).append(entry)
        per_bucket = max(1, target // len(by_key))
        picked = []
        for bucket in by_key.values():
            picked.extend(rng.sample(bucket, min(per_bucket, len(bucket))))
        return picked[:target]
    # signal_boosted: oversample negative feedback, retries, slow calls
    def weight(entry: dict) -> float:
        w = 1.0
        if entry.get("feedback") == "thumbs_down":
            w += 4.0
        if entry.get("retried"):
            w += 2.0
        if entry.get("latency_ms", 0) > 1400:
            w += 1.0
        return w
    weighted = sorted(entries,
                      key=lambda e: (weight(e), rng.random()), reverse=True)
    head = weighted[:target // 2]
    tail_pool = [e for e in entries if e not in head]
    return head + rng.sample(tail_pool, target - len(head))
