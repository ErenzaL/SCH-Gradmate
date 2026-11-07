from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .course_parser import credit_totals, normalize_code, normalize_name
from .document_loader import PROJECT_ROOT


def load_rules(path: str | Path | None = None) -> dict:
    rules_path = Path(path) if path else PROJECT_ROOT / "data" / "graduation_rules.json"
    return json.loads(rules_path.read_text(encoding="utf-8"))


def select_rule(year: int | str, rules: dict | None = None) -> dict | None:
    rules = rules or load_rules()
    y = int(year)
    if str(y) in rules and "total_credits" in rules[str(y)]:
        return rules[str(y)]
    if 2019 <= y <= 2021:
        return rules.get("2019-2021")
    if y == 2022:
        return rules.get("2022")
    if 2023 <= y <= 2026:
        return rules.get("2023-2026")
    return None


def match_courses(courses: pd.DataFrame, catalog: list[dict] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Match exact verified course names/codes only. Never guess a near match."""
    result = courses.copy()
    if catalog is None:
        catalog_path = PROJECT_ROOT / "data" / "curriculum_courses.json"
        if catalog_path.exists():
            try:
                loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
                if loaded.get("verified") is True:
                    catalog = loaded.get("courses", [])
            except (OSError, ValueError, json.JSONDecodeError):
                catalog = None
    if not catalog:
        result["curriculum_match"] = "확인 필요"
        result["official_category"] = None
        result["matched_course_name"] = None
        return result, list(dict.fromkeys(result.loc[result["course_name"].ne(""), "course_name"].tolist()))

    by_code, by_name = {}, {}
    for item in catalog:
        if item.get("verified") is False:
            continue
        canonical = item.get("course_name", "")
        code = normalize_code(item.get("course_code", ""))
        if code:
            by_code[code] = item
        for candidate in [canonical, *item.get("aliases", [])]:
            if normalize_name(candidate):
                by_name[normalize_name(candidate)] = item

    statuses, categories, canonical_names, unmatched = [], [], [], []
    for _, row in result.iterrows():
        item = by_code.get(normalize_code(row.get("course_code", ""))) if row.get("course_code", "") else None
        if item is None:
            item = by_name.get(normalize_name(row.get("course_name", "")))
        if item:
            statuses.append("일치")
            categories.append(item.get("category"))
            canonical_names.append(item.get("course_name"))
        else:
            statuses.append("미매칭")
            categories.append(None)
            canonical_names.append(None)
            if row.get("course_name", ""):
                unmatched.append(row["course_name"])
    result["curriculum_match"] = statuses
    result["official_category"] = categories
    result["matched_course_name"] = canonical_names
    return result, list(dict.fromkeys(unmatched))


def _criterion(name: str, current: float | None, required: float | None, source: dict | None, detail: str = "") -> dict:
    if current is None or required is None:
        return {"name": name, "status": "확인 필요", "detail": detail or "제공된 자료 또는 입력에서 기준을 확인할 수 없습니다.", "source": source}
    missing = max(0.0, float(required) - float(current))
    return {"name": name, "status": "충족" if missing == 0 else "미충족", "current": float(current), "required": float(required), "missing": missing, "detail": detail, "source": source}


def check_graduation_requirements(
    year: int | str,
    totals: dict[str, Any],
    courses: pd.DataFrame | None = None,
    activity_result: dict | None = None,
    rules: dict | None = None,
) -> dict[str, Any]:
    """Evaluate verified requirements only; return unknown criteria as 확인 필요."""
    rule = select_rule(year, rules)
    source = {"document": rule.get("source"), "page": rule.get("page")} if rule and rule.get("source") else None
    requirements: list[dict] = []
    if not rule:
        for name in ("총 이수학점", "전공 이수학점", "교양 이수학점"):
            requirements.append(_criterion(name, None, None, None, "해당 학번의 기준 자료가 없습니다."))
        rule = {}
        source = None
    else:
        requirements.append(_criterion("총 이수학점", totals.get("total"), rule.get("total_credits"), source))
        major_rule = rule.get("major", {})
        general_rule = rule.get("general_education", {})
        requirements.append(_criterion("전공 이수학점", totals.get("major"), major_rule.get("total"), source))
        requirements.append(_criterion("교양 이수학점", totals.get("general"), general_rule.get("total"), source))

        detail_specs = [
            ("전공기초", "major_basic", major_rule.get("basic"), "major"),
            ("전공필수 학점", "major_required", major_rule.get("required"), "major"),
            ("전공선택", "major_elective", major_rule.get("elective"), "major"),
            ("교양공통", "general_common", general_rule.get("common"), "general"),
            ("교양영역", "general_area", general_rule.get("required_area"), "general"),
            ("교양선택", "general_elective", general_rule.get("elective"), "general"),
        ]
        for label, key, required, parent in detail_specs:
            if required is None:
                continue
            unclassified = totals.get("subcategory_unclassified_credits", {}).get(parent, 0)
            if unclassified > 0:
                requirements.append(_criterion(label, None, required, source, f"{unclassified:g}학점의 {parent} 과목이 세부 이수구분으로 분류되지 않았습니다."))
            else:
                current = totals.get("detail_totals", {}).get(key, 0.0)
                requirements.append(_criterion(label, current, required, source))

    mandatory_courses = rule.get("mandatory_courses")
    if mandatory_courses:
        if courses is None:
            requirements.append(_criterion("전공필수 과목", None, None, source, "수강과목 목록을 확인할 수 없습니다."))
        else:
            matched_courses, _ = match_courses(courses)
            completed = set(matched_courses.loc[(matched_courses["status_class"] == "completed") & (matched_courses["curriculum_match"] == "일치"), "matched_course_name"])
            missing_courses = [name for name in mandatory_courses if name not in completed]
            requirements.append({"name":"전공필수 과목", "status":"충족" if not missing_courses else "미충족", "missing_courses":missing_courses, "source":source})
    else:
        requirements.append(_criterion("전공필수 과목 목록", None, None, source, "공식 과목별 필수 목록을 확인할 수 없어 자동 대조하지 않았습니다."))

    certification_points = rule.get("certification_points")
    if certification_points is None:
        requirements.append(_criterion("비교과 졸업인증", None, None, source, "학번별 비교과 점수 기준을 확인할 수 없습니다."))
    elif activity_result is None:
        requirements.append(_criterion("비교과 졸업인증", None, None, source, "비교과 활동 자료가 등록되지 않았습니다."))
    else:
        requirements.append(_criterion("비교과 졸업인증", activity_result.get("estimated_points"), certification_points, source, "학생 입력 기준 예상 점수입니다."))

    verified_other = rule.get("other_requirements")
    if verified_other:
        for requirement in verified_other:
            requirements.append(_criterion(requirement, None, None, source, "이 요건은 업로드한 과목표만으로 판정할 수 없습니다."))
    else:
        requirements.append(_criterion("봉사학습 및 기타 학과 요건", None, None, source, "수강과목표만으로는 판정할 수 없어 학사 자료 또는 수동 확인이 필요합니다."))

    unmatched = []
    matched_frame = courses
    if courses is not None:
        matched_frame, unmatched = match_courses(courses)
    all_credit = all(item["status"] == "충족" for item in requirements if item["name"] in {"총 이수학점", "전공 이수학점", "교양 이수학점"})
    return {
        "admission_year": int(year), "requirements": requirements,
        "unrecognized_courses": unmatched, "matched_courses": matched_frame,
        "source": source, "credit_requirements_satisfied": all_credit,
        "official_determination": False,
    }


def check_credits(year: int | str, totals: dict, rules: dict | None = None) -> list[dict]:
    """Backward-compatible credit-only interface used by the earlier MVP."""
    result = check_graduation_requirements(year, totals, rules=rules)
    return [item for item in result["requirements"] if item["name"] in {"총 이수학점", "전공 이수학점", "교양 이수학점"}]


def check_graduation_eligibility(
    admission_year: int | str,
    completed_courses: pd.DataFrame,
    extracurricular_activities: dict[str, Any] | None,
    student_info: dict[str, Any] | None = None,
    rules: dict | None = None,
) -> dict[str, Any]:
    """수강과목과 비교과 학생 입력을 하나의 출처 추적 가능한 결과로 묶습니다."""
    from .activity_checker import calculate_activity_points

    totals = credit_totals(completed_courses)
    activity_result = calculate_activity_points(admission_year, rules=rules, **extracurricular_activities) if extracurricular_activities is not None else None
    result = check_graduation_requirements(admission_year, totals, completed_courses, activity_result, rules=rules)
    if activity_result is not None:
        for requirement in result["requirements"]:
            if requirement["name"] == "비교과 졸업인증":
                requirement["status"] = activity_result["status"]
                requirement["detail"] = "학생 입력 예상점수 기준이며 증빙과 공식 승인 여부 확인이 필요합니다."
                requirement["reported_approved_points"] = activity_result.get("reported_approved_points")
    # 신분별 면제 기준은 명시된 규칙이 없으므로 추측하거나 면제 처리하지 않습니다.
    result["student_info"] = {key: value for key, value in (student_info or {}).items() if key in {"international_student"}}
    result["activity_result"] = activity_result
    result["course_totals"] = totals
    result["overall_status"] = (
        "확인 필요" if any(row["status"] == "확인 필요" for row in result["requirements"])
        else "미충족" if any(row["status"] == "미충족" for row in result["requirements"])
        else "충족"
    )
    result["official_determination"] = False
    return result
