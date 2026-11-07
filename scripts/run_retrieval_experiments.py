"""검색 방식 비교 실험. 실행: python scripts/run_retrieval_experiments.py

관련성 라벨은 data/evaluation/questions.json 의 (자료명, 쪽) 쌍이며, 라벨이 있는 답변 가능 질문만 지표 계산에 사용합니다.
결과는 data/evaluation/results/experiment_*.csv 로 저장되어 재현할 수 있습니다.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import retrieval as R  # noqa: E402
from src.document_loader import load_documents  # noqa: E402

RESULTS = ROOT / "data" / "evaluation" / "results"
KS = (1, 3, 5)
REPEATS = 5


def load_questions() -> list[dict]:
    questions = json.loads((ROOT / "data" / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    return [q for q in questions if q["answerable"] and q["relevant_sources"]]


def relevant_pages(q: dict) -> set[tuple[str, int]]:
    return {(unicodedata.normalize("NFC", s["source"]), s["page"]) for s in q["relevant_sources"]}


def metrics(q: dict, hits: list[dict]) -> dict:
    rel = relevant_pages(q)
    pages = [(unicodedata.normalize("NFC", h["source"]), h["page"]) for h in hits]
    out = {}
    for k in KS:
        top = pages[:k]
        found = {p for p in top if p in rel}
        out[f"recall@{k}"] = len(found) / len(rel)
        out[f"precision@{k}"] = sum(1 for p in top if p in rel) / k
        out[f"hit@{k}"] = 1.0 if found else 0.0
    rr = next((1.0 / (i + 1) for i, p in enumerate(pages[:5]) if p in rel), 0.0)
    out["mrr@5"] = rr
    return out


def run_config(name: str, questions: list[dict], fn) -> list[dict]:
    rows = []
    for q in questions:
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter(); hits = fn(q); times.append((time.perf_counter() - t) * 1000)
        rows.append({"config": name, "question_id": q["question_id"], "category": q["category"],
                     "question_type": q.get("question_type", "factual"), **metrics(q, hits),
                     "latency_ms": statistics.median(times)})
    return rows


def summarize(per_query: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = [c for c in per_query.columns if c.startswith(("recall@", "precision@", "hit@", "mrr@"))] + ["latency_ms"]
    return per_query.groupby(by)[cols].mean().round(4).reset_index()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    records, ix, bm, _ = R.index()
    dense_ok = ix is not None
    print(f"questions={len(questions)} chunks={len(records)} dense={dense_ok}")
    configs = {
        "bm25": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "bm25", use_anchors=False),
        "bm25+anchors(filter)": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "bm25", use_anchors=True),
        "bm25+anchors(promote)": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "bm25", use_anchors="promote"),
        "bm25+rerank": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "bm25", rerank=True, use_anchors=False),
    }
    if dense_ok:
        configs |= {
            "dense": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "dense", use_anchors=False),
            "hybrid": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "hybrid", use_anchors=False),
            "hybrid+anchors": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "hybrid", use_anchors=True),
            "rrf": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "rrf", use_anchors=False),
            "hybrid+rerank": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "hybrid", rerank=True, use_anchors=False),
            "hybrid+anchors+rerank": lambda q: R.retrieve(q["question"], q["admission_year"], 5, "hybrid", rerank=True, use_anchors=True),
        }
    rows = []
    for name, fn in configs.items():
        rows += run_config(name, questions, fn)
        print("done", name)
    per_query = pd.DataFrame(rows)
    per_query.to_csv(RESULTS / "experiment_retrieval_per_query.csv", index=False, encoding="utf-8-sig")
    summary = summarize(per_query, ["config"])
    summary.to_csv(RESULTS / "experiment_retrieval_summary.csv", index=False, encoding="utf-8-sig")
    summarize(per_query, ["config", "category"]).to_csv(RESULTS / "experiment_retrieval_by_category.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))

    # chunk-size / overlap experiment (rebuilds an in-memory corpus for each setting)
    size_rows = []
    for size, overlap in ((600, 80), (1100, 120), (1800, 200)):
        recs, _ = load_documents(chunk_size=size, overlap=overlap)
        corpus = R.build_corpus(recs, with_dense=dense_ok)
        # relevance is page-level, so labels remain valid for every chunk size
        methods = [("bm25", False)] + ([("hybrid", False)] if dense_ok else [])
        for method, anchors in methods:
            r = run_config(f"{method}|{size}/{overlap}", questions,
                           lambda q, m=method, c=corpus, a=anchors: R.retrieve(q["question"], q["admission_year"], 5, m, use_anchors=a, corpus=c))
            df = pd.DataFrame(r)
            row = df[[c for c in df.columns if c.startswith(("recall@", "precision@", "hit@", "mrr@"))] + ["latency_ms"]].mean().round(4).to_dict()
            size_rows.append({"method": method, "chunk_size": size, "overlap": overlap, "chunks": len(recs), **row})
    chunk_df = pd.DataFrame(size_rows)
    chunk_df.to_csv(RESULTS / "experiment_chunk_size.csv", index=False, encoding="utf-8-sig")
    print(chunk_df.to_string(index=False))


def header_ablation(questions: list[dict]) -> pd.DataFrame:
    """색인 텍스트에 문맥 헤더(자료명·학과·학번·절)를 붙였을 때와 붙이지 않았을 때를 비교합니다."""
    records, _ = load_documents()
    dense_ok = R.dense_available()
    rows = []
    for contextual in (False, True):
        corpus = R.build_corpus(records, with_dense=dense_ok, contextual=contextual)
        for method in ("bm25",) + (("dense", "hybrid", "rrf") if dense_ok else ()):
            r = pd.DataFrame(run_config(method, questions, lambda q, m=method, c=corpus: R.retrieve(
                q["question"], q["admission_year"], 5, m, use_anchors=False, corpus=c)))
            rows.append({"contextual_header": contextual, "method": method,
                         **r[[c for c in r.columns if c.startswith(("recall@", "precision@", "mrr@"))]].mean().round(4).to_dict()})
    return pd.DataFrame(rows)


def significance(per_query: pd.DataFrame, pairs: list[tuple[str, str]], metric: str = "recall@5", n: int = 10000) -> pd.DataFrame:
    """질문 단위 붓스트랩(95% 신뢰구간)으로 방식 간 차이가 우연 범위인지 확인합니다."""
    import numpy as np
    piv = per_query.pivot(index="question_id", columns="config", values=metric)
    rng = np.random.default_rng(0)
    rows = []
    for a, b in pairs:
        if a not in piv or b not in piv:
            continue
        diff = (piv[a] - piv[b]).values
        means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        rows.append({"a": a, "b": b, "metric": metric, "mean_diff": round(float(diff.mean()), 4),
                     "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                     "significant": bool(lo > 0 or hi < 0), "n_questions": len(diff)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
    qs = load_questions()
    ab = header_ablation(qs)
    ab.to_csv(RESULTS / "experiment_header_ablation.csv", index=False, encoding="utf-8-sig")
    print(ab.to_string(index=False))
    pq = pd.read_csv(RESULTS / "experiment_retrieval_per_query.csv", encoding="utf-8-sig")
    sig = significance(pq, [("hybrid", "bm25"), ("rrf", "bm25"), ("dense", "bm25"), ("rrf", "hybrid"),
                            ("hybrid+rerank", "hybrid"), ("bm25+anchors(filter)", "bm25"), ("bm25+anchors(promote)", "bm25"), ("hybrid+anchors", "hybrid"), ("bm25+rerank", "bm25")])
    sig.to_csv(RESULTS / "experiment_significance.csv", index=False, encoding="utf-8-sig")
    print(sig.to_string(index=False))
