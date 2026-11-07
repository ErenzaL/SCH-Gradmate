import pandas as pd
from src.course_parser import parse_courses, credit_totals
from src.graduation_checker import select_rule, check_credits, check_graduation_requirements, match_courses


def make_courses(rows):
    body = "course_name,credits,category,status\n" + "".join(",".join(map(str, row)) + "\n" for row in rows)
    frame, _ = parse_courses(body.encode(), "courses.csv")
    return frame


def test_admission_year_mapping_uses_verified_rule_only():
    assert select_rule(2024)["total_credits"] == 130
    assert select_rule(2022)["certification_points"] == 95
    assert select_rule(2018) is None


def test_credit_calculation_for_verified_2024_rule():
    frame = make_courses([("전공과목", 58, "전공", "completed"), ("교양과목", 32, "교양", "completed")])
    results = check_credits(2024, credit_totals(frame))
    major = next(item for item in results if item["name"] == "전공 이수학점")
    assert major["status"] == "미충족"
    assert major["missing"] == 18


def test_unknown_year_credit_requirements_are_confirm_needed():
    results = check_credits(2018, {"total":100, "major":50, "general":30})
    assert all(item["status"] == "확인 필요" for item in results)


def test_unsupported_mandatory_activity_and_other_requirements_are_not_guessed():
    frame = make_courses([("임의 과목", 3, "전공", "completed")])
    result = check_graduation_requirements(2024, credit_totals(frame), courses=frame)
    by_name = {item["name"]: item for item in result["requirements"]}
    assert by_name["전공필수 과목 목록"]["status"] == "확인 필요"
    assert by_name["비교과 졸업인증"]["status"] == "확인 필요"
    assert by_name["봉사학습 및 기타 학과 요건"]["status"] == "확인 필요"
    assert result["official_determination"] is False


def test_course_names_are_unmatched_without_a_verified_catalog():
    frame = make_courses([("가장 비슷한 과목", 3, "전공", "completed")])
    matched, unrecognized = match_courses(frame, catalog=[])
    assert unrecognized == ["가장 비슷한 과목"]
    assert matched.iloc[0]["curriculum_match"] == "확인 필요"


def test_exact_verified_course_code_or_alias_match_only():
    frame = make_courses([("선형 대수학", 3, "전공", "completed"), ("선형대수학 기초", 3, "전공", "completed")])
    catalog = [{"course_name":"선형대수학", "course_code":"CS101", "aliases":["선형 대수학"], "category":"전공필수", "verified":True}]
    matched, unmatched = match_courses(frame, catalog=catalog)
    assert matched.iloc[0]["curriculum_match"] == "일치"
    assert matched.iloc[0]["official_category"] == "전공필수"
    assert matched.iloc[1]["curriculum_match"] == "미매칭"
    assert unmatched == ["선형대수학 기초"]


def test_admission_rule_does_not_accept_wrong_cohort_source():
    rules = {"2024":{"total_credits":130,"source":"verified 2024","major":{"total":76},"general_education":{"total":32}}, "2025":{"total_credits":120,"source":"verified 2025","major":{"total":70},"general_education":{"total":30}}}
    assert select_rule(2024, rules)["source"] == "verified 2024"
    assert select_rule(2025, rules)["source"] == "verified 2025"
