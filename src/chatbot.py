"""질문 한 건을 처리하는 통합 진입점. 규칙 계산 → 검색 → 근거 평가 → (선택) LLM 생성 순서로 동작합니다."""
from __future__ import annotations

import os
import re

from . import rag
from .query_utils import MAX_QUESTION_CHARS, content_terms, extract_year, has_domain_terms, is_greeting, is_injection, normalize
from .retrieval import active_backend, index, retrieve
from .rule_engine import ASK_YEAR, answer_rule

EMPTY_PROMPT = "질문을 입력해 주세요. 예: '2024학번 졸업학점은 몇 학점인가요?'"
TOO_LONG = f"질문이 너무 깁니다. {MAX_QUESTION_CHARS}자 이내로 줄여서 다시 질문해 주세요."
INJECTION_REPLY = "그 요청은 도와드릴 수 없어요. 순천향대학교 AI·빅데이터학과 졸업요건, 다면적 졸업인증제, 봉사학습에 대해 질문해 주세요."
GREETING = "안녕하세요! 졸업학점, 다면적 졸업인증제, 봉사학습처럼 졸업요건에 대해 궁금한 점을 물어보세요. 학번을 함께 알려 주시면 더 정확해요."
_FOLLOW_UP = re.compile(r"^(그럼|그러면|그리고|또|그건|그것은|그 경우|그렇다면|거기서)\b")


def _result(text: str, mode: str, citations=None, hits=None, warnings=None, **extra) -> dict:
    return {"text": text, "mode": mode, "citations": citations or [], "hits": hits or [], "warnings": warnings or [],
            "needs_year": mode == "clarify", "evidence_level": extra.pop("evidence_level", None), "generator": extra.pop("generator", "none"), "verified_text": extra.pop("verified_text", None), **extra}


COOLING_NOTE = "AI 요약이 잠시 쉬는 중이라 이번에는 자료·표 기반 답변으로 안내합니다."


def _llm_ready(use_llm: bool) -> bool:
    return bool(use_llm and rag.gemini_configured())


def _llm_failure_note(exc: Exception) -> str:
    if isinstance(exc, rag.LLMCoolingDown):
        return COOLING_NOTE
    code = getattr(exc, "code", None)
    return "AI 요약 서비스가 응답하지 않아 자료 발췌로 안내했습니다." + (f" (오류 코드 {code})" if code is not None else "")


def _previous_user_question(history: list[dict] | None, current: str | None = None) -> str | None:
    """직전 사용자 질문. 호출부가 현재 질문까지 기록에 넣었더라도 현재 질문은 건너뜁니다."""
    skipped_current = False
    for message in reversed(history or []):
        if message.get("role") != "user" or not message.get("content"):
            continue
        if current is not None and not skipped_current and normalize(message["content"]) == current:
            skipped_current = True
            continue
        return message["content"]
    return None


def _is_elliptical(question: str) -> bool:
    return bool(_FOLLOW_UP.match(question)) or len(content_terms(question)) <= 1


def _year_variants(hits: list[dict]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        ys = hit.get("admission_years")
        if ys:
            label = f"{ys[0]}~{ys[-1]}학번" if len(ys) > 1 else f"{ys[0]}학번"
            if label not in seen:
                seen.append(label)
    return seen


def respond(question: str, year: int | None = None, history: list[dict] | None = None, use_llm: bool = True) -> dict:
    """반환: text, mode(rule|clarify|extractive|llm|insufficient|refused|empty|greeting|...), citations, hits, warnings."""
    q = normalize(question)
    if not q:
        return _result(EMPTY_PROMPT, "empty")
    if len(q) > MAX_QUESTION_CHARS:
        return _result(TOO_LONG, "empty")
    if is_injection(q):
        return _result(INJECTION_REPLY, "refused")
    try:
        index()  # 문서 어휘를 불러와 잡담·무관 질문 판별에 씁니다(이미 불러왔으면 즉시 반환)
    except Exception:
        pass
    explicit_year = extract_year(q)
    year = explicit_year or year

    previous = _previous_user_question(history, q)
    # 학번을 되묻는 질문에 "2024학번입니다"처럼 학번만 답하면 직전 질문을 그 학번으로 다시 답합니다.
    if explicit_year and previous and not has_domain_terms(q):
        q = normalize(previous)
        previous = _previous_user_question(history, q)
    rule = answer_rule(q, year)
    if rule is None and previous and _is_elliptical(q):
        rule = answer_rule(f"{q} 학점", year)
    if rule is not None:
        mode = {"clarify": "clarify", "unavailable": "insufficient"}.get(rule.mode, "rule")
        # 규칙 답변도 근거 자료를 hits 로 노출합니다(자료명·쪽·적용 학번·원문 발췌).
        evidence = [{"source": c["source"], "title": c["title"], "page": c["page"], "text": c["snippet"], "admission_years": c.get("years")} for c in rule.citations]
        text, generator, extra_warnings, verified = rule.text, "rules", [], None
        # 확정 답변을 대화체로 풀어 쓰되, 숫자·주의 사항이 하나라도 달라지면 원문을 씁니다.
        if mode == "rule" or rule.mode in ("rule_caveat", "conflict"):
            if _llm_ready(use_llm) and os.getenv("GEMINI_REPHRASE", "1") != "0":
                try:
                    friendly = rag.rephrase_rule_answer(q, year, rule.text, history)
                    if friendly:
                        text, generator, verified = friendly, "rules+gemini", rule.text
                except Exception as exc:
                    extra_warnings.append(_llm_failure_note(exc))
        return _result(text, mode, rule.citations, evidence, extra_warnings, rule_mode=rule.mode, generator=generator, verified_text=verified)

    smalltalk = is_greeting(q) or not content_terms(q)
    offtopic = not smalltalk and not has_domain_terms(q)  # 졸업 자료 어휘가 하나도 없는 질문(잡담·무관한 질문)
    if smalltalk or (offtopic and _llm_ready(use_llm)):
        if _llm_ready(use_llm):
            try:
                reply = rag.small_talk_reply(q, history)
                if reply:
                    return _result(reply, "greeting", generator="gemini")
            except Exception as exc:
                if smalltalk:
                    return _result(GREETING, "greeting", warnings=[_llm_failure_note(exc)])
        if smalltalk:
            return _result(GREETING, "greeting")
    search_q = f"{previous} {q}" if previous and _FOLLOW_UP.match(q) else q

    try:
        _, _, _, warnings = index()
        hits = retrieve(search_q, year, limit=6, method="hybrid", use_anchors="promote", collapse_duplicates=True)
    except Exception:
        return _result("자료를 불러오는 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.", "error")
    level, used = rag.assess_evidence(search_q, hits)
    if level == "none":
        return _result(rag.INSUFFICIENT, "insufficient", warnings=list(warnings), evidence_level=level)

    # 학번에 따라 답이 달라지는 자료만 근거인데 학번을 모르면 먼저 묻습니다.
    if year is None and used and all(h.get("admission_years") for h in used) and len(_year_variants(used)) > 1:
        options = ", ".join(_year_variants(used))
        return _result(f"{ASK_YEAR}\n\n(자료에는 {options} 기준이 따로 있습니다.)", "clarify", warnings=list(warnings), evidence_level=level)

    warnings = list(warnings)
    if use_llm and level == "supported" and rag.gemini_configured():
        try:
            raw = rag.generate_with_gemini(search_q, year, used, history=history)
            text, ids = rag.parse_used_sources(raw, len(used))
            if re.search(r"[가-힣]", text):
                if "확인하기 어렵습니다" in text and not ids:
                    return _result(rag.INSUFFICIENT, "insufficient", warnings=warnings, evidence_level=level)
                cited = [used[i - 1] for i in ids] or used
                citations = [rag.make_citation(h, rag.best_snippet(search_q, h)) for h in cited]
                return _result(text, "llm", citations, cited, warnings, evidence_level=level, generator="gemini")
        except Exception as exc:
            warnings.append(_llm_failure_note(exc))
    text, citations = rag.compose_extractive(search_q, used, level, year)
    return _result(text, "extractive", citations, used, warnings, evidence_level=level, generator="extractive")
