from __future__ import annotations

import io
import math
import re
import unicodedata
from typing import Any

import pandas as pd

REQUIRED = {"course_name", "credits", "category", "status"}
OPTIONAL = {"course_code"}
ALIASES = {
    "과목명": "course_name", "교과목명": "course_name",
    "학점": "credits", "이수구분": "category", "구분": "category",
    "이수상태": "status", "상태": "status",
    "과목코드": "course_code", "교과목코드": "course_code", "코드": "course_code",
}
COMPLETED_STATUSES = {"completed", "complete", "이수", "완료", "이수완료", "취득", "pass", "p"}
IN_PROGRESS_STATUSES = {"in_progress", "in progress", "수강중", "이수중", "수강", "진행중"}
CATEGORY_MAP = {
    "전공": ("major", None), "전공기초": ("major", "major_basic"),
    "전공필수": ("major", "major_required"), "전공선택": ("major", "major_elective"),
    "교양": ("general", None), "교양공통": ("general", "general_common"),
    "교양영역": ("general", "general_area"), "교양선택": ("general", "general_elective"),
    # '교양필수' alone does not establish which official required-area bucket applies.
    "교양필수": ("general", None),
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def normalize_name(value: object) -> str:
    """Unicode를 정규화하고 공백 차이를 제거한 교과목 비교 키를 만듭니다."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", _text(value))).casefold()


def normalize_code(value: object) -> str:
    return re.sub(r"[\s-]+", "", unicodedata.normalize("NFKC", _text(value))).casefold()


def _status_class(value: str) -> tuple[str | None, bool]:
    key = re.sub(r"[\s_-]+", "", value).casefold()
    retake = "재수강" in key or "retake" in key
    if key in {re.sub(r"[\s_-]+", "", item).casefold() for item in COMPLETED_STATUSES} or key in {"재수강완료", "retakecompleted", "retakecomplete"}:
        return "completed", retake
    if key in {re.sub(r"[\s_-]+", "", item).casefold() for item in IN_PROGRESS_STATUSES}:
        return "in_progress", retake
    return None, retake


def validate_course_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize and validate a course table, retaining row-level issues for review."""
    if frame.empty:
        raise ValueError("수강과목 자료에 과목 행이 없습니다.")
    frame = frame.copy()
    normalized_columns = {}
    for column in frame.columns:
        key = unicodedata.normalize("NFC", str(column)).strip()
        normalized_columns[column] = ALIASES.get(key, key.casefold())
    frame = frame.rename(columns=normalized_columns)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        labels = {"course_name":"과목명", "credits":"학점", "category":"이수구분", "status":"이수상태"}
        raise ValueError("필수 항목이 없습니다: " + ", ".join(labels.get(column, column) for column in missing))
    if "course_code" not in frame:
        frame["course_code"] = ""
    frame = frame[["course_name", "course_code", "credits", "category", "status"]].copy()
    frame["course_name"] = frame["course_name"].map(_text)
    frame["course_code"] = frame["course_code"].map(_text)
    raw_credits = frame["credits"].copy()
    numeric = pd.to_numeric(raw_credits, errors="coerce")
    frame["credits"] = numeric
    frame["category"] = frame["category"].map(_text)
    frame["status"] = frame["status"].map(_text)
    frame["normalized_name"] = frame["course_name"].map(normalize_name)
    frame["normalized_code"] = frame["course_code"].map(normalize_code)

    issues: list[list[str]] = [[] for _ in range(len(frame))]
    credit_valid: list[bool] = []
    status_classes: list[str | None] = []
    retake_flags: list[bool] = []
    category_groups: list[str | None] = []
    detail_groups: list[str | None] = []
    for pos, (_, row) in enumerate(frame.iterrows()):
        if not row["course_name"]:
            issues[pos].append("과목명이 비어 있습니다.")
        value = numeric.iloc[pos]
        is_valid = False
        if pd.isna(value):
            issues[pos].append("학점이 비어 있거나 숫자가 아닙니다.")
        else:
            try:
                is_valid = math.isfinite(float(value)) and float(value) >= 0
            except (TypeError, ValueError):
                is_valid = False
            if not is_valid:
                issues[pos].append("학점은 0 이상의 숫자여야 합니다.")
        credit_valid.append(is_valid)
        category_key = normalize_name(row["category"])
        category_info = CATEGORY_MAP.get(category_key)
        if not row["category"]:
            issues[pos].append("이수구분이 비어 있습니다.")
        elif category_info is None:
            issues[pos].append("이수구분을 공식 기준에 연결할 수 없어 분류 확인이 필요합니다.")
        category_groups.append(category_info[0] if category_info else None)
        detail_groups.append(category_info[1] if category_info else None)
        status_class, retake_flag = _status_class(row["status"])
        status_classes.append(status_class)
        retake_flags.append(retake_flag)
        if not row["status"]:
            issues[pos].append("이수상태가 비어 있습니다.")
        elif status_class is None:
            issues[pos].append("이수상태를 완료 또는 수강 중으로 확인할 수 없습니다.")
        if retake_flag:
            issues[pos].append("재수강 과목의 학점 인정 여부를 확인해야 합니다.")

    frame["credit_valid"] = credit_valid
    frame["status_class"] = status_classes
    frame["category_group"] = category_groups
    frame["detail_group"] = detail_groups
    frame["retake_flag"] = retake_flags
    frame["row_issues"] = issues
    frame["duplicate_possible"] = False
    frame["duplicate_excluded"] = False

    duplicate_keys = frame.apply(lambda row: row["normalized_code"] or row["normalized_name"], axis=1)
    frame["duplicate_key"] = duplicate_keys
    completed_positions: dict[str, list[int]] = {}
    for pos, (_, row) in enumerate(frame.iterrows()):
        key = row["duplicate_key"]
        if key and row["credit_valid"] and row["status_class"] == "completed":
            completed_positions.setdefault(key, []).append(pos)
    for positions in completed_positions.values():
        if len(positions) > 1:
            for pos in positions:
                frame.iloc[pos, frame.columns.get_loc("duplicate_possible")] = True
                frame.iloc[pos, frame.columns.get_loc("row_issues")].append("같은 교과목이 중복되어 한 번만 학점에 반영합니다.")
            for pos in positions[1:]:
                frame.iloc[pos, frame.columns.get_loc("duplicate_excluded")] = True

    warnings: list[str] = []
    for pos, row_issues in enumerate(frame["row_issues"]):
        if row_issues:
            warnings.append(f"{pos + 2}행: " + " ".join(dict.fromkeys(row_issues)))
    if any(frame["duplicate_possible"]):
        warnings.append("중복·재수강 가능 과목은 성적표와 학사 기준에서 확인해 주세요.")
    return frame, warnings


def parse_courses(data: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    name = filename.lower()
    if name.endswith(".csv"):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(data), encoding="cp949")
    elif name.endswith(".xlsx"):
        frame = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    else:
        raise ValueError("CSV 또는 XLSX 파일만 올릴 수 있습니다.")
    return validate_course_frame(frame)


def credit_totals(frame: pd.DataFrame) -> dict[str, Any]:
    if "credit_valid" not in frame or "status_class" not in frame:
        frame, _ = validate_course_frame(frame)
    eligible = frame[frame["credit_valid"] & frame["status_class"].eq("completed") & ~frame["duplicate_excluded"]]
    credits = eligible["credits"].fillna(0).astype(float)
    totals: dict[str, Any] = {
        "total": float(credits.sum()),
        "major": float(eligible.loc[eligible["category_group"].eq("major"), "credits"].fillna(0).sum()),
        "general": float(eligible.loc[eligible["category_group"].eq("general"), "credits"].fillna(0).sum()),
        "completed_courses": int(len(eligible)),
        "in_progress_courses": int(frame["status_class"].eq("in_progress").sum()),
        "invalid_courses": int((~frame["credit_valid"] | frame["status_class"].isna()).sum()),
        "duplicate_courses": int(frame["duplicate_excluded"].sum()),
        "uncategorized_credits": float(eligible.loc[eligible["category_group"].isna(), "credits"].fillna(0).sum()),
        "subcategory_unclassified_credits": {
            "major": float(eligible.loc[eligible["category_group"].eq("major") & eligible["detail_group"].isna(), "credits"].fillna(0).sum()),
            "general": float(eligible.loc[eligible["category_group"].eq("general") & eligible["detail_group"].isna(), "credits"].fillna(0).sum()),
        },
        "category_totals": {},
        "detail_totals": {},
    }
    for key, amount in eligible.groupby("category_group", dropna=False)["credits"].sum().items():
        totals["category_totals"][key if pd.notna(key) else "미분류"] = float(amount)
    for key, amount in eligible.dropna(subset=["detail_group"]).groupby("detail_group")["credits"].sum().items():
        totals["detail_totals"][key] = float(amount)
    return totals
