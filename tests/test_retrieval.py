import json

import pytest

from src import retrieval as R
from src.document_loader import load_documents


@pytest.fixture(scope="module")
def corpus():
    records, _ = load_documents()
    return R.build_corpus(records, with_dense=False)


def test_default_backend_in_tests_is_bm25_without_loading_a_model():
    assert R.dense_available() is False
    assert R.active_backend() == "bm25"
    import sys
    assert "sentence_transformers" not in sys.modules and "torch" not in sys.modules


def test_bm25_finds_topcit_evidence(corpus):
    hits = R.retrieve("TOPCIT 응시", None, 5, "bm25", corpus=corpus)
    assert hits and any("TOPCIT" in h["text"] for h in hits)
    assert all(h["method_used"] == "bm25" for h in hits)


def test_dense_methods_fall_back_to_bm25_and_say_so(corpus):
    for method in ("dense", "hybrid", "rrf"):
        hits = R.retrieve("봉사학습 신청", None, 3, method, corpus=corpus)
        assert hits and hits[0]["method_used"] == "bm25"


def test_year_filter_excludes_other_cohort_rows(corpus):
    hits = R.retrieve("다면적 졸업인증 점수소계", 2019, 10, "bm25", corpus=corpus)
    assert hits
    for h in hits:
        assert h["admission_years"] is None or 2019 in h["admission_years"]
    assert not any(h["admission_years"] == [2023, 2024] for h in hits)


def test_empty_blank_and_symbol_queries_return_nothing(corpus):
    for q in ("", "   ", None):
        assert R.retrieve(q, None, 5, "bm25", corpus=corpus) == []
    assert R.retrieve("!!! ??? ###", None, 5, "bm25", corpus=corpus) == []


def test_year_with_no_matching_rows_returns_empty_not_error(corpus):
    records = [{"text": "2024학번 130학점", "source": "a.pdf", "page": 1, "admission_years": [2024], "title": "a"}]
    c = R.build_corpus(records, with_dense=False)
    assert R.retrieve("130학점", 2019, 5, "bm25", corpus=c) == []


def test_semester_tokens_distinguish_semesters(corpus):
    assert "3학년1학기" in R.tokenize("3학년 1학기 전공")
    hits = R.retrieve("3학년 1학기 전공 과목", None, 1, "bm25", corpus=corpus)
    assert hits[0]["section"] == "3학년 1학기"


def test_collapse_duplicates_keeps_one_of_equivalent_chunks():
    records, _ = load_documents()
    c = R.build_corpus(records, with_dense=False)
    plain = R.retrieve("평가교육 신청 방법", None, 10, "bm25", corpus=c)
    collapsed = R.retrieve("평가교육 신청 방법", None, 10, "bm25", collapse_duplicates=True, corpus=c)
    assert len(collapsed) <= len(plain)


def test_rerank_reorders_without_losing_hits(corpus):
    base = R.retrieve("봉사학습 신청", None, 5, "bm25", corpus=corpus)
    rer = R.rerank_results("봉사학습 신청", base, 5)
    assert {h["chunk_id"] for h in rer} == {h["chunk_id"] for h in base}
    assert all("rerank_score" in h for h in rer)


def test_index_text_adds_context_header_but_keeps_display_text():
    rec = {"text": "교양 32학점", "title": "졸업요건", "department": "AI·빅데이터학과", "admission_years": [2019, 2025], "section": "교양"}
    assert R.index_text(rec, True).startswith("졸업요건 AI·빅데이터학과 2019~2025학번 교양")
    assert R.index_text(rec, False) == "교양 32학점"


def test_artifact_roundtrip_and_stale_detection(tmp_path, monkeypatch):
    info = R.build_artifacts(with_dense=False, index_dir=tmp_path)
    assert info["records"] > 0 and not info["dense"]
    records, warnings, from_artifact = R._load_records(tmp_path)
    assert from_artifact and len(records) == info["records"]
    # 지문이 다르면 오래된 산출물을 쓰지 않고 다시 파싱합니다.
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["fingerprint"] = "stale"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _, _, from_artifact = R._load_records(tmp_path)
    assert from_artifact is False


def test_corrupt_artifact_falls_back_to_parsing(tmp_path):
    (tmp_path / "records.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{broken", encoding="utf-8")
    records, _, from_artifact = R._load_records(tmp_path)
    assert records and from_artifact is False


def test_committed_artifact_matches_current_sources():
    """배포 저장소에 포함된 색인 산출물이 현재 자료와 일치해야 Render 에서 재파싱 없이 시작합니다."""
    _, _, from_artifact = R._load_records()
    assert from_artifact, "자료나 로더가 바뀌었습니다: python scripts/build_index.py 를 실행해 산출물을 갱신하세요."
