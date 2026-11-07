"""엔드투엔드 답변 동작 평가(LLM 미사용). 실행: python scripts/run_answer_evaluation.py

각 평가 질문에 respond(use_llm=False) 를 호출해 기대 동작(답변/학번 확인/답변 거부/충돌 표시/학번 혼용 금지)과
기대 사실 포함, 금지 사실 미포함, 인용 (자료, 쪽) 일치를 규칙으로 검사합니다. 생성형 답변의 충실성은 평가하지 않습니다.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(key, None)

from src.chatbot import respond  # noqa: E402
from src.retrieval import active_backend  # noqa: E402

RESULTS = ROOT / "data" / "evaluation" / "results"
ANSWERED = {"rule", "extractive", "llm"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fact_present(fact: str, text: str) -> bool:
    """기대 사실 확인: 사실에 숫자가 있으면 모든 숫자가, 없으면 모든 한글·영문 어절(2자 이상)이 답변에 있어야 합니다."""
    # 2019, 2024 같은 학번 숫자는 적용 학번 확인(인용·금지 사실)에서 따로 보므로 사실 확인에서는 제외합니다.
    numbers = [n for n in re.findall(r"\d+(?:\.\d+)?", fact) if not re.fullmatch(r"20\d{2}", n)]
    if numbers:
        return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text) for n in numbers)
    words = re.findall(r"[가-힣A-Za-z]{2,}", fact)
    return all(w in text for w in words)


def evaluate(q: dict) -> dict:
    t = time.perf_counter()
    r = respond(q["question"], q["admission_year"], use_llm=False)
    latency = (time.perf_counter() - t) * 1000
    text = r["text"]
    behavior = q.get("expected_behavior") or ("answer" if q["answerable"] else "insufficient")
    mode, level = r["mode"], r.get("evidence_level")
    answered = mode in ANSWERED and level != "weak"
    hedged = mode == "extractive" and level == "weak"
    refused = mode in ("insufficient", "refused")
    facts = q.get("expected_answer_facts") or []
    facts = [f for f in facts if f != "제공 자료에서 확인 불가"]
    fact_recall = (sum(1 for f in facts if fact_present(f, text)) / len(facts)) if facts else None
    forbidden_hit = [f for f in q.get("forbidden_facts", []) if f in text]
    relevant = {(norm(s["source"]), s["page"]) for s in q["relevant_sources"]}
    cited = [(norm(c["source"]), c["page"]) for c in r["citations"]]
    citation_hit = (any(c in relevant for c in cited) if relevant and behavior != "clarify_year" else None)
    citation_precision = (sum(1 for c in cited if c in relevant) / len(cited)) if relevant and cited else None
    if behavior in ("answer", "answer_with_caveat", "answer_with_conflict", "calculate"):
        passed = answered and (fact_recall in (None, 1.0)) and not forbidden_hit and (citation_hit is not False)
        if behavior == "answer_with_conflict":
            passed = passed and mode == "rule" and "다릅니다" in text or "표기 차이" in text
            passed = bool(passed) and fact_recall == 1.0
    elif behavior == "clarify_year":
        passed = mode == "clarify" and not forbidden_hit
    elif behavior == "insufficient":
        declares = any(k in text for k in ("확인하기 어렵습니다", "확인되지 않습니다"))
        passed = (refused or declares) and not forbidden_hit
    elif behavior == "no_year_mixing":
        passed = not forbidden_hit and mode != "insufficient"
    else:
        passed = False
    outcome = "answered" if answered else "hedged" if hedged else "clarify" if mode == "clarify" else "refused" if refused else mode
    return {"question_id": q["question_id"], "question_type": q.get("question_type", "factual"), "category": q["category"],
            "year": q["admission_year"], "expected_behavior": behavior, "mode": mode, "evidence_level": level, "outcome": outcome,
            "passed": bool(passed), "soft_passed": bool(passed or (behavior == "insufficient" and hedged and not forbidden_hit)), "fact_recall": fact_recall, "forbidden_hit": ";".join(forbidden_hit),
            "citation_hit": citation_hit, "citation_precision": citation_precision, "n_citations": len(cited),
            "latency_ms": round(latency, 2), "question": q["question"], "answer_preview": text[:160].replace("\n", " ")}


def main() -> None:
    questions = json.loads((ROOT / "data" / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    rows = [evaluate(q) for q in questions]
    df = pd.DataFrame(rows)
    backend = active_backend()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"answer_eval_extractive_{backend}_per_question.csv", index=False, encoding="utf-8-sig")
    by_behavior = df.groupby("expected_behavior").agg(n=("passed", "size"), passed=("passed", "sum"), pass_rate=("passed", "mean"), soft_pass_rate=("soft_passed", "mean"),
                                                       fact_recall=("fact_recall", "mean"), citation_hit=("citation_hit", "mean"),
                                                       citation_precision=("citation_precision", "mean"), median_latency_ms=("latency_ms", "median")).round(3).reset_index()
    by_behavior.to_csv(RESULTS / f"answer_eval_extractive_{backend}_summary.csv", index=False, encoding="utf-8-sig")
    overall = {"n": len(df), "passed": int(df.passed.sum()), "pass_rate": round(float(df.passed.mean()), 3), "soft_pass_rate": round(float(df.soft_passed.mean()), 3),
               "p95_latency_ms": round(float(df.latency_ms.quantile(0.95)), 2), "backend": backend}
    (RESULTS / f"answer_eval_extractive_{backend}_overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(by_behavior.to_string(index=False)); print(overall)
    print(df[~df.passed][["question_id", "expected_behavior", "outcome", "fact_recall", "forbidden_hit", "citation_hit", "question"]].to_string(index=False))


if __name__ == "__main__":
    main()
