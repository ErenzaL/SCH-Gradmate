"""검증된 구조화 규칙(data/graduation_rules.json)으로 학점·점수 질문에 결정론적으로 답합니다.

LLM이 숫자를 계산하거나 학번별 규정을 섞지 못하도록, 학번이 필요한 질문은 학번을 먼저 확인하고
규칙에 값이 없으면 추정하지 않고 '확인 불가'로 안내합니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .graduation_checker import load_rules, select_rule
from .query_utils import SUPPORTED_YEARS, parse_credit_count

ASK_YEAR = "정확한 안내를 위해 입학 학번(예: 2024학번)을 알려 주세요. 학번에 따라 졸업요건이 달라질 수 있습니다."
FOOTER_UNOFFICIAL = "공식 졸업 승인 여부를 확정하는 답변은 아니므로 학과 사무실에서 최종 확인해 주세요."
GENERIC_CAVEAT = "이 표에는 학과명과 적용 학번이 명확히 표기되어 있지 않아 AI·빅데이터학과 기준인지 확정할 수 없습니다. 학과 사무실에서 확인하는 것을 권장합니다."


@dataclass
class RuleResult:
    kind: str
    text: str
    citations: list[dict] = field(default_factory=list)
    mode: str = "rule"          # rule | rule_caveat | conflict | clarify | unavailable
    needs_year: bool = False


def _cite(source: str, page: int, snippet: str, years: list[int] | None = None, note: str | None = None) -> dict:
    from pathlib import Path
    return {"source": source, "title": Path(source).stem, "page": page, "years": years, "snippet": snippet, "note": note,
            "chunk_id": None, "origin": "structured_rules"}


def _span(years: list[int]) -> str:
    return f"{years[0]}~{years[-1]}학번" if len(years) > 1 else f"{years[0]}학번"


def _generic(rules: dict) -> dict:
    return rules.get("general_requirements", {})


def _generic_value(entries: list[dict], year: int) -> dict | None:
    return next((e for e in entries if year in e["years"]), None)


def _table(rules: dict, year: int) -> dict | None:
    rule = select_rule(year, rules)
    return rule if rule and "total_credits" in rule else None


def _structure(rules: dict, year: int) -> dict | None:
    structure = rules.get("certification_structure", {})
    for entry in structure.values():
        if isinstance(entry, dict) and year in entry.get("years", []):
            return {**entry, "source": structure["source"], "page": structure["page"]}
    return None


def _clarify(kind: str) -> RuleResult:
    return RuleResult(kind, ASK_YEAR, [], "clarify", needs_year=True)


def _unavailable(kind: str, year: int, what: str, extra: str = "") -> RuleResult:
    text = f"{year}학번의 {what}은 현재 제공된 자료만으로는 확인하기 어렵습니다. {extra}학과 사무실 또는 관련 부서에 확인하는 것을 권장합니다."
    return RuleResult(kind, text, [], "unavailable")


def _year_in_scope(year: int) -> bool:
    return year in SUPPORTED_YEARS or year in range(2016, 2019)


# ---------------------------------------------------------------- credit answers
def total_credits(year: int | None, rules: dict) -> RuleResult:
    if year is None:
        return _clarify("credit_total")
    table, gen = _table(rules, year), _generic(rules)
    if table:
        m, g = table["major"], table["general_education"]
        text = (f"**{year}학번 졸업학점은 {table['total_credits']}학점입니다.** 전공 {m['total']}학점(전공기초 {m['basic']}·전공필수 {m['required']}·전공선택 {m['elective']})과 "
                f"교양 {g['total']}학점(교양공통 {g['common']}·교양영역 {g['required_area']}·교양선택 {g['elective']})으로 구성됩니다. "
                f"학점 외에 다면적 졸업인증, 봉사학습 등 별도 요건도 있습니다.")
        cites = [_cite(table["source"], table["page"], f"이수조건: 교양 {g['total']}, 전공 {m['total']}, 졸업학점 {table['total_credits']}", [year])]
        if gen.get("total_credits_min") == table["total_credits"]:
            cites.append(_cite(gen["source"], gen["page"], f"졸업학점: {gen['total_credits_min']}학점 이상", None, GENERIC_CAVEAT))
        return RuleResult("credit_total", text, cites)
    if gen and _year_in_scope(year):
        text = (f"AI·빅데이터학과 {year}학번 전용 이수현황 표는 제공된 자료에 없습니다. 학과명이 표기되지 않은 '졸업자격요건' 표에는 졸업학점이 "
                f"**{gen['total_credits_min']}학점 이상**으로 적혀 있습니다. {GENERIC_CAVEAT}")
        return RuleResult("credit_total", text, [_cite(gen["source"], gen["page"], f"졸업학점: {gen['total_credits_min']}학점 이상", None, GENERIC_CAVEAT)], "rule_caveat")
    return _unavailable("credit_total", year, "졸업학점 기준")


def remaining_credits(year: int | None, completed: float, rules: dict) -> RuleResult:
    base = total_credits(year, rules)
    if base.mode in ("clarify", "unavailable"):
        return base
    table = _table(rules, year)
    target = table["total_credits"] if table else _generic(rules)["total_credits_min"]
    gap = target - completed
    fmt = lambda v: f"{v:g}"
    if gap > 0:
        text = (f"**{year}학번 기준 총 이수학점 {fmt(target)}학점에서 현재 {fmt(completed)}학점을 이수하셨으니 {fmt(gap)}학점이 더 필요합니다.** "
                f"(계산: {fmt(target)} − {fmt(completed)} = {fmt(gap)})")
    else:
        text = (f"**총 이수학점 기준 {fmt(target)}학점을 이미 채우셨습니다.** (계산: {fmt(completed)} − {fmt(target)} = {fmt(-gap)}학점 초과) "
                f"다만 졸업에는 전공·교양 세부 학점과 다면적 졸업인증, 봉사학습 등 다른 요건도 충족해야 합니다.")
    if not table:
        text += f" 이 기준은 학과가 표기되지 않은 자료의 값입니다. {GENERIC_CAVEAT}"
    text += " 전공·교양 등 세부 구분별 부족 학점은 수강과목 분석 기능으로 확인할 수 있습니다."
    return RuleResult("credit_remaining", text, base.citations, base.mode)


def general_credits(year: int | None, rules: dict) -> RuleResult:
    if year is None:
        return _clarify("credit_general")
    table, gen = _table(rules, year), _generic(rules)
    entry = _generic_value(gen.get("general_education", []), year)
    if table:
        g = table["general_education"]
        text = f"**{year}학번 교양 이수학점은 {g['total']}학점입니다.** 교양공통 {g['common']}·교양영역 {g['required_area']}·교양선택 {g['elective']}학점으로 구성됩니다."
        cites = [_cite(table["source"], table["page"], f"교양 공통 {g['common']}, 영역 {g['required_area']}, 선택 {g['elective']}, 합계 {g['total']}", [year])]
        if entry and entry["credits"] == g["total"]:
            cites.append(_cite(gen["source"], gen["page"], f"교양: {entry['credits']}학점 이수 ({_span(entry['years'])})", entry["years"], GENERIC_CAVEAT))
        return RuleResult("credit_general", text, cites)
    if entry:
        text = (f"학과명이 표기되지 않은 '졸업자격요건' 표에는 {_span(entry['years'])} 교양이 **{entry['credits']}학점** 이수로 적혀 있습니다. "
                f"AI·빅데이터학과 {year}학번 전용 표는 제공된 자료에 없습니다. {GENERIC_CAVEAT}")
        return RuleResult("credit_general", text, [_cite(gen["source"], gen["page"], f"교양: {entry['credits']}학점 이수 ({_span(entry['years'])})", entry["years"], GENERIC_CAVEAT)], "rule_caveat")
    covered = "·".join(_span(e["years"]) for e in gen.get("general_education", []))
    return _unavailable("credit_general", year, "교양 이수학점 기준", f"제공된 졸업자격요건 표는 {covered}만 다루고 있습니다. ")


def major_credits(year: int | None, rules: dict, part: str = "total") -> RuleResult:
    labels = {"total": "전공 이수학점", "basic": "전공기초", "required": "전공필수", "elective": "전공선택"}
    kind = f"credit_major_{part}"
    if year is None:
        return _clarify(kind)
    table = _table(rules, year)
    if part == "basic":
        return basic_credits(year, rules)
    if not table:
        return _unavailable(kind, year, f"{labels[part]} 기준")
    m = table["major"]
    value = m[part]
    text = f"**{year}학번 {labels[part]} 기준은 {value}학점입니다.**"
    if part == "required" and value == 0:
        text += " 이수조건 표의 전공필수 칸이 0으로 표시되어 있습니다. 과목별 전공필수 목록은 제공된 자료에서 확인되지 않습니다(교육과정 편성표의 과정구분 열에는 전기·전공·복수1만 표시됨)."
    if part == "total":
        text += f" (전공기초 {m['basic']}·전공필수 {m['required']}·전공선택 {m['elective']})"
    cite = _cite(table["source"], table["page"], f"전공 전기 {m['basic']}, 필수 {m['required']}, 선택 {m['elective']}, 합계 {m['total']}", [year])
    return RuleResult(kind, text, [cite])


def basic_credits(year: int, rules: dict) -> RuleResult:
    table, gen = _table(rules, year), _generic(rules)
    entry = _generic_value(gen.get("department_basic", []), year)
    gen_cite = lambda: _cite(gen["source"], gen["page"], f"학과기초: {entry['credits']}학점 이수 ({_span(entry['years'])})", entry["years"], GENERIC_CAVEAT)
    if table and entry and entry["credits"] != table["major"]["basic"]:
        text = (f"**자료마다 기준이 다릅니다.** {year}학번 이수조건 표(AI·빅데이터학과)는 전공기초 **{table['major']['basic']}학점**이고, "
                f"학과명이 표기되지 않은 '졸업자격요건' 표는 {_span(entry['years'])} 학과기초를 **{entry['credits']}학점**으로 적고 있습니다. "
                f"어느 기준이 적용되는지 이 자료만으로는 확정할 수 없으니 학과 사무실에 확인해 주세요.")
        cites = [_cite(table["source"], table["page"], f"전공 전기 {table['major']['basic']}", [year]), gen_cite()]
        return RuleResult("credit_major_basic", text, cites, "conflict")
    if table:
        return RuleResult("credit_major_basic", f"**{year}학번 전공기초 기준은 {table['major']['basic']}학점입니다.**",
                          [_cite(table["source"], table["page"], f"전공 전기 {table['major']['basic']}", [year])])
    if entry:
        text = f"학과명이 표기되지 않은 '졸업자격요건' 표에는 {_span(entry['years'])} 학과기초가 **{entry['credits']}학점**으로 적혀 있습니다. {GENERIC_CAVEAT}"
        return RuleResult("credit_major_basic", text, [gen_cite()], "rule_caveat")
    return _unavailable("credit_major_basic", year, "전공기초 기준")


# ---------------------------------------------------------------- certification
def certification_points(year: int | None, rules: dict, explain: bool = False) -> RuleResult:
    if year is None:
        return _clarify("cert")
    s = _structure(rules, year)
    if not s:
        return _unavailable("cert", year, "다면적 졸업인증 점수 기준")
    c = s["components"]
    comp = " + ".join(f"{k} {v}" for k, v in c.items())
    text = f"**{year}학번 다면적 졸업인증 합계는 {s['total']}점입니다.** 구성: {comp} = {s['total']}점."
    if explain:
        items = s["items"]
        text += (f"\n\n- 교과: {', '.join(items['교과'])}\n- 비교과지정: {', '.join(items['비교과지정'])}\n- 비교과미지정: {', '.join(items['비교과미지정'])}")
        text += ("\n\n표의 비고에는 학과활동·TOPCIT 응시가 회당 1점, 경진대회 참가가 3~10점, 수상이 10~30점으로 적혀 있습니다. "
                 "활동별 인정 점수의 세부 기준은 '다면적졸업인증제 비교과 배점 기준표'를 확인해 주세요. 점수 계산기는 '비교과 활동 체크하기'에서 사용할 수 있으며, 증빙 승인 여부는 반영되지 않는 예상치입니다.")
    cites = [_cite(s["source"], s["page"], f"{_span(s['years'])} 점수소계: {comp}, 합계 {s['total']}", s["years"])]
    if explain:
        cites.append(_cite("다면적졸업인증제 비교과 배점 기준표.pdf", 1, "활동별 역량점수와 제출증빙 기준표", None))
    return RuleResult("cert", text, cites)


def topcit_levels(rules: dict) -> RuleResult:
    """수준별 점수는 학번과 무관한 '비교과 배점 기준표' 값입니다."""
    info = rules["certification_structure"]["topcit"]
    pts = info["score_table"]["points_by_level"]
    levels = ", ".join(f"{k.split('(')[0]} {v}점" for k, v in pts.items())
    text = (f"비교과 배점 기준표(2024. 3. 1 기준)에는 TOPCIT 수준별 역량점수가 **{levels}**으로 적혀 있습니다. "
            f"다만 학과 졸업인증 표의 비고는 'TOPCIT 응시(회당 1점)'으로 달리 표기하고 있어, 어느 기준이 적용되는지는 담당 부서에 확인해 주세요.")
    return RuleResult("topcit_levels", text, [_cite(info["score_table"]["source"], info["score_table"]["page"], f"TOPCIT 수준별: {levels}", None)], "conflict")


def topcit(year: int | None, rules: dict) -> RuleResult:
    if year is None:
        return _clarify("topcit")
    info = rules.get("certification_structure", {}).get("topcit", {})
    s = _structure(rules, year)
    if not s:
        return _unavailable("topcit", year, "TOPCIT 응시 배점")
    table_cite = _cite(s["source"], s["page"], f"{_span(s['years'])} 비교과지정: " + ", ".join(s["items"]["비교과지정"]), s["years"])
    if year not in info.get("listed_in_years", []):
        text = (f"{year}학번이 포함된 학과 표({_span(s['years'])})의 비교과지정 항목에는 TOPCIT 응시가 없습니다. "
                f"다른 학번의 배점을 적용할 수 없으며, 필수 여부는 제공된 자료에서 확인하기 어렵습니다.")
        return RuleResult("topcit", text, [table_cite], "rule_caveat")
    pts = info["score_table"]["points_by_level"]
    levels = ", ".join(f"{k.split('(')[0]} {v}점" for k, v in pts.items())
    text = (f"{year}학번이 포함된 학과 표({_span(s['years'])})에는 TOPCIT 응시가 비교과지정 **{info['listed_points']}점** 항목으로 있습니다. "
            f"다만 자료에는 TOPCIT 응시가 졸업의 필수 요건인지 명시되어 있지 않아 필수 여부는 확인하기 어렵습니다.\n\n"
            f"**자료 간 표기 차이:** 학과 표의 비고는 'TOPCIT 응시(회당 1점)'이고, 비교과 배점 기준표는 수준별 점수({levels})를 적고 있습니다. "
            f"어느 기준으로 인정되는지는 다면적졸업인증제 담당 부서에 확인해 주세요.")
    cites = [table_cite,
             _cite(s["source"], s["page"], "비고: 학과활동(회당 1점), TOPCIT 응시(회당 1점)", None),
             _cite(info["score_table"]["source"], info["score_table"]["page"], f"TOPCIT 수준별: {levels}", None)]
    return RuleResult("topcit", text, cites, "conflict")


# ---------------------------------------------------------------- other generic-table facts
def double_major(rules: dict) -> RuleResult:
    gen, dm = _generic(rules), _generic(rules)["double_major"]
    text = (f"학과명이 표기되지 않은 '졸업자격요건' 표에는 복수전공이 **{dm['detail']}**로 적혀 있습니다. "
            f"{GENERIC_CAVEAT} 복수전공의 신청 시기·절차는 제공된 자료에서 확인되지 않습니다.")
    return RuleResult("double_major", text, [_cite(gen["source"], gen["page"], f"복수전공: {dm['detail']}", None, GENERIC_CAVEAT)], "rule_caveat")


def thesis(rules: dict, asked_project: bool = False) -> RuleResult:
    gen = _generic(rules)
    lead = "제공된 자료에는 '졸업작품'이라는 항목이 없어 졸업작품의 필수 여부는 확인하기 어렵습니다. 다만 " if asked_project else ""
    text = f"{lead}학과명이 표기되지 않은 '졸업자격요건' 표에는 4학년에 **졸업논문 심의에 통과하여야 함**이라고 적혀 있습니다. {GENERIC_CAVEAT}"
    return RuleResult("thesis", text, [_cite(gen["source"], gen["page"], f"4학년: {gen['thesis']}", None, GENERIC_CAVEAT)], "rule_caveat")


def requirements_summary(year: int | None, rules: dict) -> RuleResult:
    if year is None:
        return _clarify("summary")
    parts, cites = [], []
    total = total_credits(year, rules)
    parts.append(total.text)
    cites += total.citations
    cert = certification_points(year, rules)
    if cert.mode != "unavailable":
        parts.append(cert.text)
        cites += cert.citations
    parts.append("봉사학습 등 다른 요건은 '졸업에 봉사활동이 필요한가요?'처럼 질문해 주세요.")
    return RuleResult("summary", "\n\n".join(parts), cites, total.mode if total.mode != "rule" else "rule")


# ---------------------------------------------------------------- dispatcher
_HOW_MUCH = re.compile(r"얼마|몇\s*학점|몇\s*점|몇\s*시간|더\s*필요|남았|부족|모자라|채워야")
_REMAIN = re.compile(r"얼마나\s*더|더\s*필요|남았|남은|부족|모자라|채워야|더\s*들어야|더\s*이수")
# 활동 항목별 점수를 묻는 질문은 학번별 합계 규칙이 아니라 배점 기준표(검색)로 답합니다.
_ITEM_WORDS = re.compile(r"기준표|설명|자격증|한국사|경진|대회|공모전|입상|수상|멘토링|동아리|특강|학술제|수준별|레벨|튜터링|스터디|연수|인증서")
_SCHEDULE = re.compile(r"언제|시기|기간|절차|방법|어떻게\s*신청|신청")


def answer_rule(question: str, year: int | None, rules: dict | None = None) -> RuleResult | None:
    """규칙 기반으로 답할 수 있는 질문이면 RuleResult, 아니면 None(검색 기반 답변으로 넘김)."""
    rules = rules or load_rules()
    q = re.sub(r"\s+", " ", question or "").casefold()
    completed = parse_credit_count(question)
    if completed is not None and _REMAIN.search(q):
        return remaining_credits(year, completed, rules)
    if "topcit" in q:
        return topcit_levels(rules) if re.search(r"수준|레벨|등급|단계", q) else topcit(year, rules)
    if any(w in q for w in ("비교과", "다면적", "졸업인증")) and re.search(r"점수|몇\s*점|합계|계산|배점|구성|필요", q) and not _ITEM_WORDS.search(q):
        return certification_points(year, rules, explain=bool(re.search(r"계산|어떻게|구성|배점", q)))
    if "복수전공" in q and not _SCHEDULE.search(q):
        return double_major(rules)
    if re.search(r"졸업\s*(논문|작품)", q):
        return thesis(rules, asked_project=bool(re.search(r"졸업\s*작품", q)))
    if "전공필수" in q:
        return major_credits(year, rules, "required")
    if "전공기초" in q or "학과기초" in q:
        return major_credits(year, rules, "basic")
    if "전공선택" in q:
        return major_credits(year, rules, "elective")
    if "교양" in q and re.search(r"학점|이수|얼마|몇", q) and not re.search(r"과목|강의|추천", q):
        return general_credits(year, rules)
    if "전공" in q and re.search(r"학점|이수|얼마|몇", q) and not re.search(r"과목|강의|추천|복수|부전공", q):
        return major_credits(year, rules, "total")
    if re.search(r"졸업\s*학점|총\s*이수\s*학점|졸업에\s*필요한\s*학점|졸업하려면\s*몇\s*학점", q):
        return total_credits(year, rules)
    if re.search(r"졸업\s*요건|졸업\s*조건|졸업\s*자격", q) and not re.search(r"봉사|논문|인증", q):
        return requirements_summary(year, rules)
    return None
