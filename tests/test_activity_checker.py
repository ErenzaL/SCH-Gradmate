import pandas as pd

from src.activity_checker import calculate_activity_points, select_activity_rule
from src.course_parser import validate_course_frame
from src.graduation_checker import check_graduation_eligibility


def activity_rules():
    return {"2024": {"verified":True, "extracurricular": {"required_points": 20, "source_ref": {"document":"verified.pdf", "page":2}, "activities":[
        {"name":"TOPCIT 응시", "points_per_count":5, "max_count":1},
        {"name":"대회 참가", "points_per_count":10, "max_count":2},
    ]}}}


def test_year_specific_activity_rules_and_unsupported_year():
    assert select_activity_rule(2024, activity_rules())["required_points"] == 20
    assert select_activity_rule(2025, activity_rules()) is None
    assert calculate_activity_points(2025, rules=activity_rules())["status"] == "확인 필요"


def test_repeatable_and_one_time_activity_caps():
    result = calculate_activity_points(2024, {"TOPCIT 응시":3, "대회 참가":8}, rules=activity_rules())
    assert result["estimated_points"] == 25
    assert result["details"][0]["recognized_count"] == 1
    assert result["details"][1]["recognized_count"] == 2
    assert result["status"] == "확인 필요"  # 예상점수만으로 충족 확정 금지


def test_missing_approval_keeps_sufficient_reported_points_provisional():
    result = calculate_activity_points(2024, {"대회 참가":2}, rules=activity_rules())
    assert result["estimated_points"] == 20
    assert result["reported_approved_points"] is None
    assert result["status"] == "확인 필요"


def test_missing_activity_rules_do_not_invent_scores():
    assert calculate_activity_points(2024, {"임의 활동":10}, rules={})["estimated_points"] is None


def test_combined_course_and_activity_check_preserves_source_and_status():
    frame, _ = validate_course_frame(pd.DataFrame([{"course_name":"자료구조", "credits":3, "category":"전공", "status":"completed"}]))
    rules = {"2024":{"verified":True,"total_credits":130,"major":{"total":76},"general_education":{"total":32},"source":"req.pdf","page":1,
                      "extracurricular":{"required_points":20,"source_ref":{"document":"cert.pdf","page":2},"activities":[{"name":"대회 참가","points_per_count":10,"max_count":2}]}}}
    result = check_graduation_eligibility(2024, frame, {"selections":{"대회 참가":1}}, rules=rules)
    by_name = {row["name"]: row for row in result["requirements"]}
    assert result["course_totals"]["major"] == 3
    assert by_name["비교과 졸업인증"]["status"] == "미충족"
    assert result["activity_result"]["source"] == {"document":"cert.pdf", "page":2}
    assert result["official_determination"] is False


def test_foreign_student_status_does_not_invent_volunteer_exemption():
    frame, _ = validate_course_frame(pd.DataFrame([{"course_name":"자료구조", "credits":3, "category":"전공", "status":"completed"}]))
    result = check_graduation_eligibility(2024, frame, None, {"international_student":True})
    volunteer = next(row for row in result["requirements"] if "봉사학습" in row["name"])
    assert volunteer["status"] == "확인 필요"
    assert result["official_determination"] is False
