import pytest

from src.graduation_checker import load_rules
from src.rule_engine import answer_rule


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def ask(rules, q, year):
    return answer_rule(q, year, rules)


def test_total_credits_2024_from_verified_table(rules):
    r = ask(rules, "졸업학점은 몇 학점인가요?", 2024)
    assert "130학점" in r.text and r.mode == "rule"
    assert r.citations[0]["source"] == "2024학번 졸업요건.jpeg" and r.citations[0]["page"] == 1


@pytest.mark.parametrize("q", ["졸업학점은 몇 학점인가요?", "교양학점은 얼마나 이수해야 하나요?", "비교과 점수는 어떻게 계산하나요?", "TOPCIT 응시는 필수인가요?", "졸업요건 알려줘"])
def test_year_required_questions_ask_for_year_instead_of_guessing(rules, q):
    r = ask(rules, q, None)
    assert r.mode == "clarify" and r.needs_year and not r.citations


def test_other_cohort_total_uses_unlabeled_table_with_caveat_not_2024_values(rules):
    r = ask(rules, "졸업학점은 몇 학점인가요?", 2023)
    assert r.mode == "rule_caveat" and "130학점 이상" in r.text and "학과명" in r.text
    assert "76학점" not in r.text and r.citations[0]["source"] == "졸업자격요건.png"


def test_cohort_outside_documented_range_is_unavailable(rules):
    r = ask(rules, "교양학점은 얼마나 이수해야 하나요?", 2026)
    assert r.mode == "unavailable" and "32학점" not in r.text.split("확인하기 어렵습니다")[0]


def test_remaining_credits_is_deterministic(rules):
    r = ask(rules, "현재 110학점을 이수했다면 얼마나 더 필요한가요?", 2024)
    assert "20학점이 더 필요" in r.text and "130 − 110 = 20" in r.text
    over = ask(rules, "현재 140학점을 이수했다면 얼마나 더 필요한가요?", 2024)
    assert "이미 채우셨습니다" in over.text and "10학점 초과" in over.text


def test_remaining_credits_needs_year(rules):
    assert ask(rules, "현재 110학점을 이수했다면 얼마나 더 필요한가요?", None).mode == "clarify"


def test_major_basic_conflict_between_sources_is_shown(rules):
    r = ask(rules, "전공기초는 몇 학점인가요?", 2024)
    assert r.mode == "conflict" and "28학점" in r.text and "22학점" in r.text and len(r.citations) == 2


def test_topcit_does_not_apply_other_cohort_points(rules):
    r2019 = ask(rules, "TOPCIT 응시는 필수인가요?", 2019)
    assert "5점" not in r2019.text and "없습니다" in r2019.text
    r2024 = ask(rules, "TOPCIT 응시는 필수인가요?", 2024)
    assert "필수 여부는 확인하기 어렵습니다" in r2024.text and r2024.mode == "conflict"


def test_topcit_levels_do_not_need_year(rules):
    r = ask(rules, "TOPCIT 수준별 점수는?", None)
    assert "수준5" in r.text and "7점" in r.text and not r.needs_year


@pytest.mark.parametrize("year,total", [(2019, 80), (2021, 80), (2022, 95), (2024, 95), (2026, 95)])
def test_certification_totals_by_cohort(rules, year, total):
    assert f"{total}점" in ask(rules, "다면적 졸업인증은 몇 점이 필요한가요?", year).text


def test_certification_explanation_lists_components(rules):
    r = ask(rules, "비교과 점수는 어떻게 계산하나요?", 2024)
    assert "대학공통 45 + 교과 5 + 비교과지정 26 + 비교과미지정 19 = 95점" in r.text


def test_item_level_questions_are_left_to_retrieval(rules):
    assert ask(rules, "한국사능력검정시험은 비교과 몇 점인가요?", None) is None
    assert ask(rules, "국제대회 입상은 비교과 몇 점인가요?", None) is None


def test_double_major_and_thesis_carry_department_caveat(rules):
    for q in ("복수전공을 하면 어떤 조건이 달라지나요?", "졸업논문이 필요한가요?"):
        r = ask(rules, q, 2024)
        assert r.mode == "rule_caveat" and "학과명" in r.text and r.citations[0]["source"] == "졸업자격요건.png"
    assert "34학점" in ask(rules, "복수전공 조건은?", 2024).text


def test_double_major_schedule_question_is_not_answered_from_credit_table(rules):
    assert ask(rules, "복수전공 신청은 언제 할 수 있나요?", 2024) is None


def test_required_major_courses_are_not_invented(rules):
    r = ask(rules, "전공필수 과목은 무엇인가요?", 2024)
    assert "0학점" in r.text and "목록은 제공된 자료에서 확인되지 않습니다" in r.text


def test_graduation_project_question_does_not_pretend_it_means_thesis(rules):
    r = ask(rules, "졸업 작품은 필수인가?", 2024)
    assert r.mode == "rule_caveat" and "'졸업작품'이라는 항목이 없어" in r.text and "졸업논문 심의" in r.text
    assert r.citations[0]["source"] == "졸업자격요건.png"
    assert "'졸업작품'이라는 항목이 없어" not in ask(rules, "졸업 논문이 필요한가요?", 2024).text
