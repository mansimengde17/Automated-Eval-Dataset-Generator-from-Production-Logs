# Automated Eval Dataset Generator from Production Logs

A system that mines production LLM logs, finds the interesting, edge case,
and failure mode interactions, and converts them into labeled evaluation
test cases automatically. The eval suite grows every night from real
traffic instead of going stale in a hand curated golden file.

Live demo: https://mansimengde17.github.io/Automated-Eval-Dataset-Generator-from-Production-Logs/

## Why this exists

The hardest part of AI evaluation is not the harness, it is the dataset.
Hand curated golden sets are expensive to build and go stale the moment
traffic shifts. This pipeline solves the data supply problem: production
traffic in, deduplicated and labeled test cases out, with humans reviewing
only the labels the judges could not agree on.

## Pipeline

```
production logs
   -> normalize + PII redaction + dedup          (ingest.py)
   -> sampling: uniform / stratified / signal boosted
   -> semantic clustering + outlier detection     (cluster.py)
   -> anomaly flags: retries, thumbs down, injection attempts
   -> multi run judge labeling + difficulty + golden refs (labeler.py)
   -> near duplicate check against existing dataset
   -> high confidence -> dataset | low confidence -> review queue
   -> eval runner + regression detector           (runner.py)
```

- `src/evalgen/logsim.py` deterministic production traffic generator so
  the whole pipeline runs offline and reproducibly
- `src/evalgen/ingest.py` schema validation, regex PII redaction, exact
  dedup, three sampling strategies
- `src/evalgen/cluster.py` bag of words embeddings, greedy density
  clustering, outlier and anomaly detection
- `src/evalgen/labeler.py` judge scoring across multiple runs, difficulty
  estimation, expected behavior labels, reference answers and rubrics,
  confidence based routing, cosine dedup at 0.92
- `src/evalgen/runner.py` eval harness with per category and per
  difficulty pass rates plus run to run regression detection
- `src/evalgen/store.py` SQLite dataset, review queue, run history
- `src/evalgen/api.py` FastAPI endpoints for the whole lifecycle

## Quick start

```bash
pip install -r requirements.txt
python demo.py                       # two pipeline nights + regression catch
python -m unittest discover tests
uvicorn evalgen.api:app --app-dir src --port 8000
```

Key endpoints:

| Route | Purpose |
|-------|---------|
| `POST /v1/pipeline/run` | run the full nightly pipeline |
| `GET /v1/dataset` | active test cases and growth stats |
| `GET /v1/review-queue` | low confidence labels awaiting a human |
| `POST /v1/review-queue/{id}` | approve, edit, or reject a label |
| `POST /v1/eval/run` | score a model and compare to the previous run |
| `GET /v1/analytics` | dataset growth and recent run history |

## Docker

```bash
docker compose up --build
```
