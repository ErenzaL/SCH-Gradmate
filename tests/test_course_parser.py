from io import BytesIO
import pandas as pd
import pytest
from src.course_parser import parse_courses, credit_totals, validate_course_frame


def test_successful_korean_csv_and_completed_credit_totals():
    raw = "course_name,credits,category,status\n선형대수학,3,전공,completed\n머신러닝,3,전공,completed\n현대기본영어,5,교양,completed\n현재수강,3,전공,in_progress\n".encode()
    frame, warnings = parse_courses(raw, "sample.csv")
    totals = credit_totals(frame)
    assert not warnings
    assert totals["total"] == 11
    assert totals["major"] == 6
    assert totals["general"] == 5
    assert totals["in_progress_courses"] == 1


def test_xlsx_parsing():
    buffer = BytesIO()
    pd.DataFrame([{"course_name":"자료구조", "credits":3, "category":"전공", "status":"completed"}]).to_excel(buffer, index=False)
    frame, _ = parse_courses(buffer.getvalue(), "courses.xlsx")
    assert credit_totals(frame)["major"] == 3


def test_missing_columns():
    with pytest.raises(ValueError, match="필수 항목"):
        parse_courses(b"course_name,credits\na,3", "bad.csv")


def test_missing_credit_is_reported_and_excluded():
    frame, warnings = parse_courses("course_name,credits,category,status\n선형대수학,,전공,completed\n".encode(), "missing.csv")
    assert any("학점이 비어" in item for item in warnings)
    assert credit_totals(frame)["total"] == 0


def test_invalid_credit_is_reported_and_excluded():
    frame, warnings = parse_courses("course_name,credits,category,status\n선형대수학,-2,전공,completed\n다른과목,abc,교양,completed\n".encode(), "invalid.csv")
    assert any("0 이상의 숫자" in item for item in warnings)
    assert credit_totals(frame)["total"] == 0


def test_duplicate_and_retaken_courses_are_not_double_counted():
    raw = "course_name,credits,category,status\n선형대수학,3,전공,completed\n선형 대수학,3,전공,재수강완료\n".encode()
    frame, warnings = parse_courses(raw, "duplicates.csv")
    assert any("중복" in item for item in warnings)
    assert any("재수강" in issue for issues in frame["row_issues"] for issue in issues)
    assert credit_totals(frame)["total"] == 3
    assert credit_totals(frame)["duplicate_courses"] == 1


def test_unknown_category_does_not_get_assumed_to_be_major_or_general():
    frame, warnings = validate_course_frame(pd.DataFrame([{"course_name":"선형대수학", "credits":3, "category":"선택", "status":"completed"}]))
    totals = credit_totals(frame)
    assert any("분류 확인" in item for item in warnings)
    assert totals["total"] == 3
    assert totals["major"] == totals["general"] == 0
    assert totals["uncategorized_credits"] == 3


def test_course_names_normalize_unicode_spacing_and_optional_code():
    raw = "과목코드,과목명,학점,이수구분,이수상태\nCS101,선형 대수학,3,전공,완료\n".encode()
    frame, _ = parse_courses(raw, "course_code.csv")
    assert frame.iloc[0]["normalized_name"] == "선형대수학"
    assert frame.iloc[0]["course_code"] == "CS101"
