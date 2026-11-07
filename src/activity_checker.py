from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_loader import PROJECT_ROOT


def load_activity_rules(path: str | Path | None = None) -> dict:
    """졸업요건 JSON에 기록된 비교과 기준을 불러옵니다."""
    rules_path = Path(path) if path else PROJECT_ROOT / "data" / "graduation_rules.json"
    return json.loads(rules_path.read_text(encoding="utf-8"))


def select_activity_rule(year: int | str, rules: dict | None = None) -> dict | None:
    y = int(year)
    all_rules = rules if rules is not None else load_activity_rules()
    exact = all_rules.get(str(y), {})
    if exact.get("verified") is True and exact.get("extracurricular"):
        return exact["extracurricular"]
    for key, years in (("2019-2021", range(2019, 2022)), ("2023-2026", range(2023, 2027))):
        if y in years:
            entry = all_rules.get(key, {})
            if entry.get("verified") is True and entry.get("extracurricular"):
                return entry["extracurricular"]
    return None


def calculate_activity_points(
    year: int | str,
    selections: dict[str, int | bool] | None = None,
    approved_points: dict[str, float] | None = None,
    rules: dict | None = None,
) -> dict[str, Any]:
    """학생 입력 예상점수와 학생이 보고한 승인점수를 별도로 합산합니다."""
    rule = select_activity_rule(year, rules)
    if not rule:
        return {"status": "확인 필요", "estimated_points": None, "reported_approved_points": None,
                "details": [], "missing_activities": [], "source": None,
                "note": "선택한 학번의 세부 비교과 배점 자료를 확인할 수 없습니다."}
    selections, approved_points = selections or {}, approved_points or {}
    details, missing, estimated, approved_sum = [], [], 0.0, 0.0
    has_approved_input = False
    for activity in rule.get("activities", []):
        name = activity["name"]
        count = max(0, int(selections.get(name, 0) or 0))
        maximum = activity.get("max_count")
        recognized_count = min(count, maximum) if maximum is not None else count
        raw = recognized_count * activity["points_per_count"]
        cap = activity.get("max_points")
        points = min(raw, cap) if cap is not None else raw
        reported_approved = max(0.0, float(approved_points.get(name, 0) or 0))
        has_approved_input |= reported_approved > 0
        estimated += points
        approved_sum += reported_approved
        if activity.get("mandatory") and count == 0:
            missing.append(name)
        details.append({"name": name, "count": count, "recognized_count": recognized_count,
                        "estimated_points": points, "reported_approved_points": reported_approved,
                        "source": rule.get("source_ref")})
    target = rule.get("required_points")
    # 입력된 실적만으로 공식 충족을 확정하지 않습니다.
    status = "확인 필요" if target is None else ("미충족" if estimated < target else "확인 필요")
    return {"status": status, "required_points": target, "estimated_points": estimated,
            "reported_approved_points": approved_sum if has_approved_input else None,
            "required_label": rule.get("required_label"), "total_certification_points": rule.get("total_certification_points"),
            "missing_activities": missing, "details": details,
            "source": rule.get("source_ref"),
            "note": "예상 점수와 승인 점수 입력은 학생 보고값입니다. 증빙 및 학사 승인 확인 전에는 충족으로 확정하지 않습니다."}


def estimate_points(year: int, checked: list[str], counts: dict[str, int] | None = None) -> dict:
    """기존 호출부와 호환되는 예상 점수 함수."""
    selections = counts or {name: int(name in checked) for name in checked}
    result = calculate_activity_points(year, selections)
    result["reported_activities"] = checked
    return result
