from __future__ import annotations

import re
import pandas as pd
import streamlit as st

from src.activity_checker import calculate_activity_points, select_activity_rule
from src.course_parser import credit_totals, parse_courses, validate_course_frame
from src.graduation_checker import check_graduation_eligibility
from src.chatbot import respond
from src.query_utils import MAX_QUESTION_CHARS, extract_year
from src.rag import gemini_explain_analysis
from src.retrieval import active_backend, index

APP_TITLE = "SCH 졸업 도우미 AI"
WELCOME = "안녕하세요! 😊\n순천향대학교 AI·빅데이터학과 졸업 도우미입니다.\n졸업요건에 대해 무엇이든 물어보세요."
SUGGESTIONS = ("졸업요건 질문하기", "수강과목 분석하기", "비교과 활동 체크하기")
QUESTION_CHIPS = ("졸업학점은 몇 학점인가요?", "교양학점은 얼마나 이수해야 하나요?", "비교과 점수는 어떻게 계산하나요?",
                  "졸업에 봉사활동이 필요한가요?", "TOPCIT 응시는 필수인가요?", "복수전공을 하면 어떤 조건이 달라지나요?")
MODE_LABEL = {"rule": "검증된 졸업요건 표 기반 계산 · AI가 만든 문장이 아닙니다",
              "extractive": "제공 자료에서 발췌한 원문입니다 · AI가 만든 문장이 아닙니다",
              "llm": "AI가 아래 근거 자료를 요약한 답변입니다 · 근거 자료를 함께 확인해 주세요"}
UPLOAD_LIMIT_BYTES = 5 * 1024 * 1024
GENERIC_FILE_ERROR = "파일을 읽지 못했어요. CSV 또는 XLSX 형식과 필수 항목(과목명, 학점, 이수구분, 이수상태)을 확인해 주세요."

st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="centered",
                   menu_items={"About": "순천향대학교 AI·빅데이터학과 졸업요건 안내 챗봇입니다. 공식 졸업 승인 여부를 확정하지 않습니다."})


@st.cache_resource(show_spinner="졸업 자료를 불러오는 중이에요...")
def load_knowledge_base() -> dict:
    """서버 프로세스당 한 번만 색인을 불러옵니다(Streamlit 재실행마다 다시 만들지 않음)."""
    records, _, _, warnings = index()
    return {"chunks": len(records), "backend": active_backend(), "warnings": list(warnings),
            "sources": sorted({r["source"] for r in records})}


def safe_error_text(exc: Exception) -> str:
    """사용자에게는 우리가 만든 한국어 안내만 보여 주고 라이브러리 오류 문구는 숨깁니다."""
    message = str(exc)
    return message if isinstance(exc, ValueError) and re.search(r"[가-힣]", message) and "Traceback" not in message and len(message) < 200 else GENERIC_FILE_ERROR

st.markdown("""
<style>
:root { --sch-blue:#0878C9; --sch-ink:#182433; --sch-muted:#64748b; --sch-line:#e6ebf0; }
.stApp { background: #f5f7fa; color:var(--sch-ink); }
[data-testid="stHeader"] { background:#f5f7fa; }
footer { visibility:hidden; }
[data-testid="stMainBlockContainer"] { max-width: 760px; padding-top: 4.75rem; padding-bottom: 5.5rem; }
.sch-header { background:#fff; border:1px solid var(--sch-line); border-radius:18px; padding:16px 20px; margin-bottom:20px; box-shadow:0 3px 14px rgba(18,39,62,.04); }
.sch-title { font-size:1.25rem; line-height:1.35; font-weight:750; color:#142333; margin:0; }
.sch-subtitle { font-size:.88rem; color:var(--sch-muted); margin-top:3px; }
.sch-welcome { background:#fff; border:1px solid var(--sch-line); border-radius:20px; padding:22px 24px; margin:4px 0 16px; box-shadow:0 5px 22px rgba(18,39,62,.045); }
.sch-welcome p { white-space:pre-line; font-size:1.08rem; line-height:1.75; margin:0; }
div.stButton > button { border-radius:13px; border:1px solid #dce5ed; background:#fff; color:#243447; min-height:45px; transition:all .15s ease; }
div.stButton > button:hover { border-color:var(--sch-blue); color:var(--sch-blue); background:#f4faff; }
[data-testid="stChatMessage"] { border:1px solid var(--sch-line); border-radius:18px; padding:13px 16px; margin:10px 0; box-shadow:0 2px 9px rgba(18,39,62,.035); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height:1.7; }
[data-testid="stChatMessage-user"] { background:#eaf4fc; border-color:#d5eafa; }
[data-testid="stChatMessage-assistant"] { background:#fff; }
[data-testid="stChatInput"] { border-radius:18px; }
[data-testid="stChatInput"] textarea { border-radius:16px; }
[data-testid="stFileUploader"] { background:#fff; border-radius:14px; }
[data-testid="stFileUploaderDropzoneInstructions"] > span:first-child { font-size:0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > span:first-child::after { content:"파일을 여기에 끌어 놓아 주세요"; font-size:.88rem; }
[data-testid="stFileUploaderDropzoneInstructions"] > small { font-size:0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > small::after { content:"CSV 또는 XLSX 파일을 선택할 수 있습니다"; font-size:.76rem; }
[data-testid="stFileUploaderDropzone"] button { font-size:0 !important; }
[data-testid="stFileUploaderDropzone"] button::after { content:"파일 선택"; font-size:.85rem; }
[data-testid="stForm"] { border:1px solid var(--sch-line); background:#fff; border-radius:16px; padding:16px; }
.stCaption { color:var(--sch-muted); }
@media (max-width:640px) {
 [data-testid="stMainBlockContainer"] { padding: 4.5rem .75rem 5.25rem; }
 .sch-header { padding:13px 15px; border-radius:15px; }
 .sch-welcome { padding:18px; border-radius:16px; }
 .sch-title { font-size:1.12rem; }
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tool" not in st.session_state:
    st.session_state.active_tool = None
if "course_totals" not in st.session_state:
    st.session_state.course_totals = None
if "course_frame" not in st.session_state:
    st.session_state.course_frame = None
if "course_review_needed" not in st.session_state:
    st.session_state.course_review_needed = False
if "course_analysis_pending" not in st.session_state:
    st.session_state.course_analysis_pending = False
if "course_warnings" not in st.session_state:
    st.session_state.course_warnings = []
if "activity_result" not in st.session_state:
    st.session_state.activity_result = None
if "activity_inputs" not in st.session_state:
    st.session_state.activity_inputs = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "admission_year" not in st.session_state:
    st.session_state.admission_year = None
if "year_selector_revision" not in st.session_state:
    st.session_state.year_selector_revision = 0

header_left, header_year, header_reset = st.columns([5.2, 2.5, 1.25], vertical_alignment="center")
with header_left:
    st.markdown(f'<div class="sch-title">{APP_TITLE}</div><div class="sch-subtitle">순천향대학교 AI·빅데이터학과</div>', unsafe_allow_html=True)
with header_year:
    selected_year = st.selectbox(
        "입학 학번",
        [None, *range(2019, 2027)],
        index=0 if st.session_state.admission_year is None else int(st.session_state.admission_year) - 2018,
        format_func=lambda value: "학번 선택" if value is None else f"{value}학번",
        label_visibility="collapsed",
        key=f"year_selector_{st.session_state.year_selector_revision}",
    )
    st.session_state.admission_year = selected_year
with header_reset:
    if st.button("새 대화", help="대화 내용 지우기", width="stretch"):
        st.session_state.year_selector_revision += 1
        for key in ("messages", "course_totals", "course_frame", "course_review_needed", "course_analysis_pending", "course_warnings", "activity_result", "activity_inputs", "active_tool", "pending_prompt", "admission_year"):
            st.session_state[key] = [] if key in ("messages", "course_warnings") else None
        st.rerun()


def add_message(role: str, content: str, **meta) -> None:
    st.session_state.messages.append({"role": role, "content": content, **meta})


def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|~])", r"\\\1", text)


def show_answer(message: dict | str) -> None:
    """답변 본문, 생성 방식 표시, 근거 자료(자료명·쪽·적용 학번·원문 발췌)를 구분해 보여 줍니다."""
    if isinstance(message, str):
        message = {"content": message}
    answer_lines, sources = [], []
    for line in message["content"].splitlines():
        if line.strip().startswith("출처:"):
            sources.append(line.strip().removeprefix("출처:").strip())
        else:
            answer_lines.append(line)
    st.markdown("\n".join(answer_lines).strip())
    label = MODE_LABEL.get(message.get("mode"))
    if label:
        st.caption(label)
    citations = message.get("citations") or []
    if citations:
        with st.expander(f"근거 자료 {len(citations)}건 보기"):
            for c in citations:
                years = c.get("years")
                span = f" · {years[0]}~{years[-1]}학번" if years and len(years) > 1 else (f" · {years[0]}학번" if years else " · 적용 학번 미표기")
                date = f" · 기준일 {c['document_date']}" if c.get("document_date") else ""
                tag = " · 수기 전사본" if c.get("transcribed") or c.get("origin") == "structured_rules" else ""
                st.markdown(f"**{_md_escape(c['title'])}** {c['page']}쪽{span}{date}{tag}")
                snippet = (c.get("snippet") or "").strip()
                if snippet:
                    st.markdown("> " + _md_escape(snippet).replace("\n", "  \n> "))
                if c.get("note"):
                    st.caption(c["note"])
    elif sources:
        st.caption("출처  ·  " + "  ·  ".join(sources))


def credit_report(year: int, totals: dict, courses: pd.DataFrame | None = None) -> str:
    course_frame = courses if courses is not None else st.session_state.course_frame
    if course_frame is not None:
        unified = check_graduation_eligibility(year, course_frame, st.session_state.activity_inputs)
        result = {"requirements": unified["requirements"], "unrecognized_courses": unified["unrecognized_courses"], "source": unified["source"]}
    else:
        result = {"requirements": [], "unrecognized_courses": [], "source": None}
        for name, field in (("총 이수학점", "total_credits"), ("전공 이수학점", "major"), ("교양 이수학점", "general")):
            result["requirements"].append({"name":name, "status":"확인 필요", "detail":"과목 파일을 등록하면 확인할 수 있습니다."})
    lines = [
        f"**수강과목 분석 결과 · {year}학번 기준**",
        f"현재 이수학점: 총 {totals['total']:g}학점 · 전공 {totals['major']:g}학점 · 교양 {totals['general']:g}학점",
    ]
    if totals.get("in_progress_courses"):
        lines.append(f"수강 중인 {totals['in_progress_courses']}개 과목은 이수학점 합계에서 제외했습니다.")
    if totals.get("duplicate_courses"):
        lines.append(f"중복 가능 과목 {totals['duplicate_courses']}개는 중복 계산에서 제외했습니다. 재수강 인정 내역을 확인해 주세요.")
    for item in result["requirements"]:
        if item["status"] in ("충족", "미충족") and "missing" in item:
            suffix = f" · {item['missing']:g}학점 부족" if item["missing"] else ""
            lines.append(f"- {item['name']}: {item['status']} ({item['current']:g}/{item['required']:g}학점{suffix})")
        elif item["status"] in ("충족", "미충족") and item.get("missing_courses") is not None:
            missing = ", ".join(item["missing_courses"]) or "없음"
            lines.append(f"- {item['name']}: {item['status']} (확인 대상: {missing})")
        else:
            lines.append(f"- {item['name']}: 확인 필요 — {item.get('detail', '공식 자료나 추가 입력을 확인해 주세요.')}")
    activity = st.session_state.activity_result
    if activity:
        target = activity.get("required_points")
        current = activity.get("estimated_points")
        if target is None or current is None:
            lines.append("- 비교과 졸업인증: 확인 필요")
        else:
            scope = activity.get("required_label") or "예상"
            lines.append(f"- 비교과 졸업인증: {activity['status']} (학생 입력 {scope} {current:g}/{target:g}점)")
            if activity.get("missing_activities"):
                lines.append("  - 입력되지 않은 필수 활동: " + ", ".join(activity["missing_activities"]))
            if activity.get("reported_approved_points") is not None:
                lines.append(f"  - 본인 입력 승인 점수: {activity['reported_approved_points']:g}점 (증빙 확인 전 참고값)")
    else:
        lines.append("- 비교과 졸업인증: 확인 필요 (활동 체크 필요)")
    if result["unrecognized_courses"]:
        names = ", ".join(result["unrecognized_courses"][:8])
        lines.append(f"\n과목명 대조 확인 필요: {names}{' 외' if len(result['unrecognized_courses']) > 8 else ''}")
    if result["source"]:
        lines.append(f"\n출처: {result['source']['document']} {result['source']['page']}쪽")
    lines.append("\n수강과목표와 제공된 기준에 따른 참고 분석이며, 공식 졸업 승인 여부를 확정하지 않습니다.")
    return "\n".join(lines)


def explain_python_result(year: int, report: str) -> str:
    # Gemini에는 식별 정보·개별 과목명·활동별 내역을 보내지 않고 필요한 합계와 상태만 전달합니다.
    safe_lines = []
    requirement_status = []
    for line in report.splitlines():
        if line.startswith("- "):
            match = re.match(r"- ((?:총 이수학점|전공 이수학점|교양 이수학점|비교과 졸업인증)[^:]*): ([^ (]+)", line)
            if match:
                requirement_status.append(f"{match.group(1)} {match.group(2)}")
            continue
        if "과목명 대조 확인 필요" in line or "입력되지 않은 필수 활동" in line:
            continue
        safe_lines.append(line)
    if requirement_status:
        safe_lines.append("요건 상태 요약: " + ", ".join(requirement_status))
    explanation = gemini_explain_analysis("현재 학번 기준으로 계산 결과를 이해하기 쉽게 설명해 주세요.", "\n".join(safe_lines), year)
    return f"{explanation}\n\n{report}" if explanation else report


if st.session_state.admission_year is not None and st.session_state.course_analysis_pending and st.session_state.course_totals is not None:
    add_message("assistant", credit_report(st.session_state.admission_year, st.session_state.course_totals, st.session_state.course_frame))
    st.session_state.course_analysis_pending = False


def answer_question(question: str) -> str | dict:
    explicit = extract_year(question)
    year = explicit if explicit else st.session_state.admission_year
    if explicit and 2019 <= year <= 2026:
        st.session_state.admission_year = year
        st.session_state.year_selector_revision += 1
    if st.session_state.course_analysis_pending and year is None:
        return "정확한 졸업요건 분석을 위해 몇 학번인지 알려주세요."
    if st.session_state.course_analysis_pending and year is not None:
        st.session_state.course_analysis_pending = False
        report = credit_report(year, st.session_state.course_totals, st.session_state.course_frame)
        return explain_python_result(year, report)
    course_terms = ("제 수강", "내 수강", "수강과목", "졸업 가능", "졸업할 수", "부족 학점", "몇 학점 남", "전공필수", "필수 과목", "필요한 수업", "교양학점", "이수학점", "충족 여부", "전체 졸업요건")
    if st.session_state.course_totals and any(term in question for term in course_terms):
        if year is None:
            return "정확한 졸업요건 분석을 위해 몇 학번인지 알려주세요."
        report = credit_report(year, st.session_state.course_totals, st.session_state.course_frame)
        return explain_python_result(year, report)
    if st.session_state.activity_result and any(term in question.casefold() for term in ("비교과 점수", "점수 부족", "비교과 얼마나", "topcit", "어떤 비교과", "활동 더", "승인 점수")):
        if year is None:
            return "활동 내역은 저장했어요. 필요한 비교과 기준을 확인하려면 입학 학번을 알려주세요."
        target = st.session_state.activity_result.get("required_points")
        points = st.session_state.activity_result.get("estimated_points")
        if target is None:
            return "선택하신 학번의 비교과 점수 기준은 현재 자료에서 확인하기 어렵습니다."
        gap = max(0, target - points)
        status = "미충족" if gap else "확인 필요"
        detail_rows = []
        for row in st.session_state.activity_result.get("details", []):
            if row["count"]:
                cap_note = f" (입력 {row['count']}회 중 인정 상한 {row['recognized_count']}회)" if row["count"] > row["recognized_count"] else ""
                detail_rows.append(f"- {row['name']}: {row['estimated_points']:g}점{cap_note}")
        detail = "\n".join(detail_rows) or "선택된 활동이 없습니다."
        source = st.session_state.activity_result.get("source")
        citation = f"\n\n출처: {source['document']} {source['page']}쪽" if source else ""
        report = f"학생 입력 예상 점수는 {points:g}점이며, {year}학번 기준 {target}점까지 {gap:g}점 부족합니다. 판정: {status}.\n\n{detail}\n\n{st.session_state.activity_result['note']}{citation}"
        return explain_python_result(year, report)
    try:
        load_knowledge_base()
        result = respond(question, year, history=st.session_state.messages[:-1][-6:])  # 방금 추가한 현재 질문은 제외
    except Exception:
        return {"content": "답변을 준비하는 중 문제가 생겼어요. 잠시 후 다시 질문해 주세요.", "mode": "error", "citations": []}
    return {"content": result["text"], "mode": result["mode"], "citations": result["citations"], "warnings": result["warnings"]}


# 첫 방문에만 간결한 인사와 빠른 시작 버튼을 보여줍니다.
welcome_slot = st.empty()
if not st.session_state.messages and st.session_state.active_tool is None:
    with welcome_slot.container():
        st.markdown(f'<div class="sch-welcome"><p>{WELCOME}</p></div>', unsafe_allow_html=True)
        suggestion_cols = st.columns(3)
        for col, label in zip(suggestion_cols, SUGGESTIONS):
            if col.button(label, key=f"suggestion_{label}", width="stretch"):
                if label == "졸업요건 질문하기":
                    st.session_state.pending_prompt = "졸업요건을 알려주세요."
                    add_message("user", "졸업요건 질문하기")
                elif label == "수강과목 분석하기":
                    st.session_state.active_tool = "course"
                    add_message("user", "수강과목 분석하기")
                    add_message("assistant", "수강과목 파일을 올리면 이수 학점과 학번별 기준을 비교해 드릴게요.")
                else:
                    st.session_state.active_tool = "activity"
                    add_message("user", "비교과 활동 체크하기")
                    add_message("assistant", "완료한 비교과 활동을 선택해 주세요. 입력하신 내용으로 예상 점수를 확인해 드릴게요.")
                st.rerun()
        st.caption("자주 묻는 질문")
        for row in (QUESTION_CHIPS[:3], QUESTION_CHIPS[3:]):
            for col, text in zip(st.columns(3), row):
                if col.button(text, key=f"chip_{text}", width="stretch"):
                    st.session_state.pending_prompt = text
                    st.rerun()

# 이전 대화 기록
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="😊" if message["role"] == "user" else "🎓"):
        show_answer(message)

# 도구는 별도 페이지가 아니라 대화 영역 안에서 펼쳐집니다.
if st.session_state.active_tool == "course":
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown("CSV 또는 XLSX 수강과목 파일을 선택해 주세요.")
        st.download_button("예시 양식 받기", "course_name,credits,category,status\n선형대수학,3,전공,completed\n머신러닝,3,전공,completed\n현대기본영어,5,교양,completed\n", "수강과목_양식.csv", "text/csv")
        with st.form("course_upload_form", clear_on_submit=False):
            uploaded = st.file_uploader("수강과목 파일", type=["csv", "xlsx"], label_visibility="collapsed", key="course_upload")
            st.caption("필수 항목: 과목명(course_name), 학점(credits), 이수구분(category), 이수상태(status). 과목코드는 선택 항목입니다.")
            submitted = st.form_submit_button("수강과목 분석하기", type="primary", width="stretch")
        if submitted:
            if not uploaded:
                st.warning("먼저 수강과목 파일을 선택해 주세요.")
            elif uploaded.size > UPLOAD_LIMIT_BYTES:
                st.error("파일이 너무 큽니다. 5MB 이하의 CSV 또는 XLSX 파일을 올려 주세요.")
            else:
                try:
                    courses, warnings = parse_courses(uploaded.getvalue(), uploaded.name)
                    totals = credit_totals(courses)
                    st.session_state.course_totals = totals
                    st.session_state.course_warnings = warnings
                    st.session_state.course_frame = courses
                    st.session_state.course_review_needed = True
                    if st.session_state.admission_year:
                        report = credit_report(st.session_state.admission_year, totals, courses)
                        add_message("assistant", report)
                    else:
                        st.session_state.course_analysis_pending = True
                        add_message("assistant", "정확한 졸업요건 분석을 위해 몇 학번인지 알려주세요.")
                    if warnings:
                        add_message("assistant", "파일에서 확인할 항목이 있습니다. 아래 과목 목록에서 수정한 뒤 다시 확인해 주세요.\n\n" + "\n".join(f"- {warning}" for warning in warnings))
                    st.session_state.active_tool = None
                    st.rerun()
                except Exception as exc:
                    st.error(safe_error_text(exc))

if st.session_state.course_review_needed and st.session_state.course_frame is not None:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown("과목명·이수구분·학점이 잘못 읽힌 항목은 아래에서 수정할 수 있습니다. 미등록 과목을 임의로 다른 과목에 연결하지 않습니다.")
        editable = st.session_state.course_frame[["course_name", "course_code", "credits", "category", "status"]].rename(columns={
            "course_name": "과목명", "course_code": "과목코드(선택)", "credits": "학점", "category": "이수구분", "status": "이수상태"
        })
        edited = st.data_editor(editable, hide_index=True, width="stretch", num_rows="dynamic", key="course_editor")
        if st.button("수정한 과목으로 다시 확인", key="recheck_courses", type="primary", width="stretch"):
            try:
                canonical = edited.rename(columns={"과목명":"course_name", "과목코드(선택)":"course_code", "학점":"credits", "이수구분":"category", "이수상태":"status"})
                courses, warnings = validate_course_frame(canonical)
                totals = credit_totals(courses)
                st.session_state.course_frame = courses
                st.session_state.course_totals = totals
                st.session_state.course_warnings = warnings
                if st.session_state.admission_year:
                    report = credit_report(st.session_state.admission_year, totals, courses)
                else:
                    st.session_state.course_analysis_pending = True
                    report = "정확한 졸업요건 분석을 위해 몇 학번인지 알려주세요."
                if warnings:
                    report += "\n\n수정 후에도 확인할 항목:\n" + "\n".join(f"- {warning}" for warning in warnings)
                add_message("assistant", report)
                st.session_state.course_review_needed = False
                st.rerun()
            except Exception as exc:
                st.error(safe_error_text(exc))

if st.session_state.active_tool == "activity":
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown("해당 학번 자료에 기재된 활동만 표시합니다. 예상 배점과 본인 입력 승인 점수는 참고용이며, 증빙 승인 여부를 대신하지 않습니다.")
        if not st.session_state.admission_year:
            st.info("활동 기준을 적용하려면 먼저 입학 학번을 선택해 주세요.")
        else:
            year = st.session_state.admission_year
            rule = select_activity_rule(year)
            if not rule:
                st.info("선택한 학번의 활동별 배점 자료는 확인되지 않았습니다. 추정 점수를 계산하지 않습니다.")
            else:
                with st.form("activity_form"):
                    selections, approved = {}, {}
                    for activity in rule.get("activities", []):
                        name = activity["name"]
                        max_count = activity.get("max_count")
                        if max_count == 1:
                            count = int(st.checkbox(name, key=f"act_{name}"))
                        else:
                            count = st.number_input(f"{name} 참여 횟수", 0, int(max_count) if max_count else 99, 0, key=f"act_{name}")
                        selections[name] = int(count)
                        cap = activity.get("max_points")
                        point_label = f"{activity['points_per_count']}점/회" + (f" · 최대 {cap}점" if cap is not None else "")
                        st.caption(f"예상 배점: {point_label}")
                        approved[name] = st.number_input(f"{name} 공식 승인 점수(본인 입력)", min_value=0.0, value=0.0, step=1.0, key=f"approved_{name}")
                    st.caption("공식 승인 점수 입력은 학생 보고값으로 기록되며, 시스템에서 증빙을 검증하지 않습니다.")
                    submitted_activity = st.form_submit_button("활동 점수 확인", type="primary", width="stretch")
                if submitted_activity:
                    inputs = {"selections": selections, "approved_points": approved}
                    result = calculate_activity_points(year, **inputs)
                    st.session_state.activity_inputs = inputs
                    st.session_state.activity_result = result
                    target = result.get("required_points")
                    gap = max(0, target - result["estimated_points"]) if target is not None else None
                    report = ["**비교과 활동 점검 결과**", f"학생 입력 예상 점수: {result['estimated_points']:g}점", f"점수 기준: {target}점" if target is not None else "점수 기준: 확인 필요", f"부족 점수(예상치 기준): {gap:g}점" if gap is not None else "부족 점수: 확인 필요", f"판정: {result['status']}", result["note"]]
                    if result.get("missing_activities"):
                        report.append("입력되지 않은 필수 활동: " + ", ".join(result["missing_activities"]))
                    if result.get("reported_approved_points") is not None:
                        report.append(f"본인 입력 승인 점수 합계: {result['reported_approved_points']:g}점 (미검증)")
                    if result.get("source"):
                        report.append(f"출처: {result['source']['document']} {result['source']['page']}쪽")
                    report_text = "\n\n".join(report)
                    add_message("assistant", explain_python_result(year, report_text))
                    st.session_state.active_tool = None
                    st.rerun()

# 자료 현황·한계·개인정보 안내는 사이드바에 모아 둡니다.
with st.sidebar:
    st.markdown("### 안내")
    try:
        status = load_knowledge_base()
        from src.rag import gemini_configured
        st.caption(f"불러온 자료 조각 {status['chunks']}개 · 검색 방식 {'키워드+의미(하이브리드)' if status['backend'] == 'hybrid' else '키워드(BM25)'} · AI 요약 {'켜짐' if gemini_configured() else '꺼짐(자료 원문 발췌)'}")
        with st.expander("사용한 자료"):
            for name in status["sources"]:
                st.markdown(f"- {name}")
        if status["warnings"]:
            with st.expander("자료 처리 한계"):
                st.markdown("\n".join(f"- {w}" for w in status["warnings"]))
    except Exception:
        st.caption("자료를 불러오지 못했어요. 잠시 후 새로고침해 주세요.")
    st.caption("공식 졸업 승인 여부를 확정하는 도구가 아닙니다. 학번별 규정은 학과 사무실에서 최종 확인해 주세요.")
    st.caption("AI 요약이 켜져 있으면 질문과 검색된 자료 일부가 Google Gemini API로 전송됩니다. 개인정보는 입력하지 마세요.")

# 입력창은 항상 대화 흐름의 아래쪽에 둡니다.
prompt = st.chat_input("졸업요건에 대해 궁금한 점을 입력하세요.", max_chars=MAX_QUESTION_CHARS)
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
if prompt:
    welcome_slot.empty()
    add_message("user", prompt)
    with st.chat_message("user", avatar="😊"):
        st.markdown(prompt)
    if prompt.strip() in ("수강과목 분석하기", "비교과 활동 체크하기"):
        st.session_state.active_tool = "course" if prompt.strip() == "수강과목 분석하기" else "activity"
        add_message("assistant", "수강과목 파일을 선택해 주세요." if st.session_state.active_tool == "course" else "완료한 비교과 활동을 선택해 주세요.")
        st.rerun()
    else:
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("자료를 살펴보고 답변을 준비하고 있어요..."):
                response = answer_question(prompt)
            if isinstance(response, str):
                response = {"content": response}
            show_answer(response)
        add_message("assistant", **response)
