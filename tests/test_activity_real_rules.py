"""실제 data/graduation_rules.json 의 비교과 규칙으로 계산기를 검사합니다(합성 규칙이 아님)."""
import pandas as pd

from src.activity_checker import calculate_activity_points, select_activity_rule
from src.course_parser import validate_course_frame
from src.graduation_checker import check_graduation_eligibility


def everything(year):
    rule = select_activity_rule(year)
    return {a["name"]: a["max_count"] for a in rule["activities"]}


def test_maximum_input_reaches_but_never_exceeds_the_designated_subtotal():
    for year, subtotal in ((2020, 11), (2022, 21), (2023, 26), (2025, 26), (2026, 26)):
        result = calculate_activity_points(year, {k: v + 5 for k, v in everything(year).items()})  # 상한 초과 입력
        assert result["estimated_points"] == subtotal, year
        assert result["required_points"] == subtotal and result["status"] == "확인 필요"  # 충족 확정 금지


def test_partial_input_is_marked_unmet_and_scope_is_stated():
    result = calculate_activity_points(2024, {"학과활동": 3})
    assert result["estimated_points"] == 3 and result["status"] == "미충족"
    assert "비교과지정" in result["required_label"] and result["total_certification_points"] == 95


def test_topcit_not_available_for_cohorts_whose_table_has_no_topcit():
    assert "TOPCIT 응시" not in everything(2021) and "TOPCIT 응시" not in everything(2022)
    assert "TOPCIT 응시" in everything(2024)


def test_unsupported_cohort_has_no_rule_and_no_invented_points():
    assert select_activity_rule(2018) is None
    assert calculate_activity_points(2018, {"학과활동": 5})["estimated_points"] is None


def test_unified_check_compares_with_subtotal_not_with_95():
    frame, _ = validate_course_frame(pd.DataFrame([{"course_name": "자료구조", "credits": 3, "category": "전공", "status": "completed"}]))
    result = check_graduation_eligibility(2024, frame, {"selections": {k: v for k, v in everything(2024).items()}, "approved_points": {}})
    row = next(r for r in result["requirements"] if r["name"] == "비교과 졸업인증")
    assert row["required"] == 26 and row["current"] == 26 and "전체 졸업인증 합계 95점" in row["detail"]
