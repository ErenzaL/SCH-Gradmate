"""구조화 규칙·평가 데이터의 내부 일관성 검사. 출처에 없는 값이 섞이면 실패합니다."""
import json
import re
import unicodedata
from pathlib import Path

import pytest

from src.document_loader import PROJECT_ROOT, load_documents
from src.graduation_checker import load_rules

RULES = load_rules()
DOCS = {unicodedata.normalize("NFC", p.name) for p in (PROJECT_ROOT / "data" / "documents").iterdir() if p.is_file()}


def test_certification_components_sum_to_total_and_designated_items_sum_to_subtotal():
    for key, s in RULES["certification_structure"].items():
        if not isinstance(s, dict) or "components" not in s:
            continue
        assert sum(s["components"].values()) == s["total"], key
        designated = sum(int(re.search(r"\((\d+)\)$", item).group(1)) for item in s["items"]["비교과지정"])
        assert designated == s["components"]["비교과지정"], key
        course = sum(int(re.search(r"\((\d+)\)$", item).group(1)) for item in s["items"]["교과"])
        assert course == s["components"]["교과"], key


def test_activity_rules_cannot_award_more_than_the_designated_subtotal():
    """과거 데이터는 활동 최대 합이 81점이었으나 표의 비교과지정 소계는 26점이었습니다."""
    for key in ("2019-2021", "2022", "2023-2026"):
        ex = RULES[key]["extracurricular"]
        maximum = sum(min(a["points_per_count"] * a["max_count"], a.get("max_points", 10**9)) for a in ex["activities"])
        assert maximum == ex["required_points"], (key, maximum)
        assert ex["total_certification_points"] == RULES[key]["certification_points"]


def test_2024_credit_table_is_internally_consistent():
    r = RULES["2024"]
    assert r["major"]["basic"] + r["major"]["required"] + r["major"]["elective"] == r["major"]["total"]
    assert sum(r["general_education"][k] for k in ("common", "required_area", "elective")) == r["general_education"]["total"]
    assert r["major"]["total"] + r["general_education"]["total"] <= r["total_credits"]


def test_every_rule_source_file_exists_in_documents_folder():
    sources = {v["source"] for v in RULES.values() if isinstance(v, dict) and "source" in v}
    sources |= {RULES["certification_structure"]["source"], RULES["general_requirements"]["source"]}
    for s in sources:
        assert unicodedata.normalize("NFC", s) in DOCS, s


def test_unlabeled_department_table_is_never_marked_confirmed():
    assert RULES["general_requirements"]["department_confirmed"] is False


def test_transcriptions_reference_existing_sources_and_have_no_empty_text():
    data = json.loads((PROJECT_ROOT / "data" / "image_transcriptions.json").read_text(encoding="utf-8"))
    assert data["_meta"]["verification_status"] == "unverified_by_human"
    ids = [r["id"] for r in data["records"]]
    assert len(ids) == len(set(ids))
    for r in data["records"]:
        assert unicodedata.normalize("NFC", r["source"]) in DOCS and r["text"].strip()


def test_curriculum_semester_credit_totals_match_printed_subtotals():
    """전사 검산: 학년별 소계(28, 29, 41, 41 → 총 139)와 전사한 학점이 일치해야 합니다."""
    records, _ = load_documents()
    curr = [r for r in records if r["source"] == "AI빅데이터학과 교육과정.pdf" and r["section"] and "학년" in r["section"]]
    by_year = {}
    for r in curr:
        year = int(r["section"][0])
        # 복수전공 칸의 졸업및취창업지도(복)은 인쇄된 소계에 포함됨(4학년 41학점)
        by_year[year] = by_year.get(year, 0) + sum(int(n) for n in re.findall(r"(\d+)학점", re.sub(r"\d학년 소계.*|총계.*", "", r["text"])))
    assert by_year == {1: 28, 2: 29, 3: 41, 4: 41}


QUESTIONS = json.loads((PROJECT_ROOT / "data" / "evaluation" / "questions.json").read_text(encoding="utf-8"))


def test_evaluation_dataset_schema_and_labels_are_valid():
    ids = [q["question_id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)) >= 50
    chunk_ids = {r["chunk_id"] for r in load_documents()[0]}
    kinds = {q.get("question_type", "factual") for q in QUESTIONS}
    assert {"unanswerable", "conflicting", "ambiguous_year", "calculation", "injection", "year_trap"} <= kinds
    for q in QUESTIONS:
        for key in ("question", "category", "admission_year", "expected_answer_facts", "relevant_sources", "answerable"):
            assert key in q, (q["question_id"], key)
        if q["answerable"]:
            assert q["relevant_sources"], q["question_id"]
        for s in q["relevant_sources"]:
            assert unicodedata.normalize("NFC", s["source"]) in DOCS
            assert s["chunk_ids"] and set(s["chunk_ids"]) <= chunk_ids, (q["question_id"], s["source"], s["page"])


def test_original_questions_are_preserved():
    original = json.loads((PROJECT_ROOT / "data" / "evaluation" / "questions_v1_original.json").read_text(encoding="utf-8"))
    current = {q["question_id"]: q for q in QUESTIONS}
    for q in original:
        assert q["question_id"] in current and current[q["question_id"]]["question"] == q["question"]
