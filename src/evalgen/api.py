"""FastAPI surface for the eval dataset generator."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .logsim import generate_logs
from .pipeline import run_pipeline
from .runner import detect_regressions, run_eval
from .store import DatasetStore

app = FastAPI(title="Eval Dataset Generator", version="1.0.0")
store = DatasetStore()


class PipelineRequest(BaseModel):
    sample_mode: str = "signal_boosted"
    sample_target: int = 200
    log_count: int = 600
    log_seed: int = 7


class ReviewDecision(BaseModel):
    decision: str  # approve | edit | reject
    reviewer: str
    edited: dict | None = None


class EvalRequest(BaseModel):
    model: str


@app.post("/v1/pipeline/run")
def pipeline(request: PipelineRequest):
    logs = generate_logs(request.log_count, request.log_seed)
    return run_pipeline(logs, store, request.sample_mode,
                        request.sample_target)


@app.get("/v1/dataset")
def dataset():
    return {"cases": store.dataset(), "stats": store.growth_stats()}


@app.get("/v1/review-queue")
def review_queue():
    return store.review_queue()


@app.post("/v1/review-queue/{case_id}")
def resolve(case_id: str, decision: ReviewDecision):
    store.resolve_review(case_id, decision.decision, decision.reviewer,
                         decision.edited)
    return {"case_id": case_id, "decision": decision.decision}


@app.post("/v1/eval/run")
def eval_run(request: EvalRequest):
    cases = store.dataset()
    if not cases:
        raise HTTPException(400, "dataset is empty; run the pipeline first")
    report = run_eval(request.model, cases)
    run_id = store.save_run(request.model, report)
    runs = store.last_runs(2)
    regression = None
    if len(runs) == 2:
        regression = detect_regressions(runs[1]["report"],
                                        runs[0]["report"])
    return {"run_id": run_id, "report": report, "regression": regression}


@app.get("/v1/analytics")
def analytics():
    return {"dataset": store.growth_stats(),
            "recent_runs": [{"run_id": r["run_id"], "model": r["model"],
                             "pass_rate": r["report"]["pass_rate"]}
                            for r in store.last_runs(10)]}
