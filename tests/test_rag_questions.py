import re
from src.rag import answer

QUESTIONS = [
    ("2024학번 졸업학점은 몇 학점인가요?", 2024),
    ("전공필수 과목은 무엇인가요?", 2024),
    ("TOPCIT 응시는 졸업 필수인가요?", 2024),
    ("비교과 졸업인증은 몇 점이 필요한가요?", 2024),
    ("봉사학습을 어떻게 신청하나요?", 2024),
    ("외국인 유학생도 봉사학습을 이수해야 하나요?", 2024),
]


def test_six_requested_questions_have_relevant_korean_evidence_and_citations():
    results = []
    for question, year in QUESTIONS:
        response, hits, _ = answer(question, year)
        assert hits, f"근거 문서를 찾지 못했습니다: {question}"
        assert "출처:" in response and re.search(r"\d+쪽", response), response
        assert any(hit.get("admission_years") is None or year in hit["admission_years"] for hit in hits)
        assert "현재 제공된 자료만으로는 확인하기 어렵습니다" not in response
        results.append(response)
    assert "130학점" in results[0]
    assert "전공필수" in results[1]
    assert "필수 여부는 확인하기 어렵습니다" in results[2]
    assert "95점" in results[3]
    assert "봉사학습교육신청" in results[4]
    assert "외국인" in results[5] and "제외" in results[5]


def test_admission_year_filter_excludes_other_cohort_rows():
    _, _, warnings = answer("TOPCIT 응시는 졸업 필수인가요?", 2019)
    from src.retrieval import search
    hits, _ = search("TOPCIT 응시는 졸업 필수인가요?", 2019)
    assert all(hit.get("admission_years") is None or 2019 in hit["admission_years"] for hit in hits)
    assert not any("AI·빅데이터학과_20240305.pdf" in hit["source"] and hit.get("admission_years") == [2023, 2024] for hit in hits)


def test_unsupported_question_returns_insufficient_evidence_message():
    response, hits, _ = answer("학과 축제는 몇 시에 시작하나요?", 2024)
    assert "현재 제공된 자료만으로는 확인하기 어렵습니다." in response
    assert hits == []


def test_shared_application_keyword_does_not_answer_unrelated_dormitory_question():
    response, hits, _ = answer("순천향대 기숙사 신청 마감일이 언제인가요?", 2024)
    assert "현재 제공된 자료만으로는 확인하기 어렵습니다." in response
    assert hits == []


def test_gemini_api_warnings_do_not_mutate_cached_document_warnings(monkeypatch):
    import src.rag as rag
    import google.genai

    cached_warnings = ["문서 인식 안내"]
    monkeypatch.setattr(rag, "search", lambda question, year: ([{"source":"자료.pdf", "page":1, "text":"졸업요건"}], cached_warnings))
    monkeypatch.setattr(rag, "_has_answer_evidence", lambda question, hits: True)
    monkeypatch.setattr(rag, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag, "_configured_secret", lambda key: {"GEMINI_API_KEY":"테스트키", "GEMINI_MODEL":"시험모델"}.get(key))

    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("test failure")

    class FailingClient:
        def __init__(self, **kwargs):
            self.models = FailingModels()

    monkeypatch.setattr(google.genai, "Client", FailingClient)
    _, _, returned_warnings = rag.gemini_answer("졸업요건을 알려주세요", 2024)
    assert cached_warnings == ["문서 인식 안내"]
    assert any("Gemini 답변 생성 서비스가 응답하지 않아" in warning for warning in returned_warnings)
