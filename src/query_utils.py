"""질문 분석 도구: 학번 추출, 핵심어 추출, 근거 포함도 계산, 프롬프트 주입 탐지."""
from __future__ import annotations

import re
import unicodedata

SUPPORTED_YEARS = range(2019, 2027)
MAX_QUESTION_CHARS = 500

_YEAR_FULL = re.compile(r"(?<!\d)(20\d{2})\s*학번")
_YEAR_SHORT = re.compile(r"(?<!\d)(\d{2})\s*학번")

# 질문의 주제를 말해 주지 않는 일반 어휘와 어미
GENERIC_TERMS = {
    "학과", "졸업", "요건", "기준", "관련", "무엇", "어떻게", "알려", "알려주세요", "알려줘", "주세요", "필요", "필요한가요",
    "되나요", "있나요", "인가요", "하나요", "한가요", "몇", "얼마", "얼마나", "언제", "어디", "어느", "경우", "때", "내용",
    "자료", "정보", "대해", "대한", "하면", "해야", "할까요", "있는", "있어요", "어떤", "무슨", "점이", "것이", "수", "있을까요",
    "이전", "지시", "무시", "시작하나요", "시작", "그럼", "그러면", "그리고", "또", "저는", "제가", "제", "내", "나", "현재",
    "순천향대", "순천향대학교", "됐어요", "됐어", "됐다", "했어요", "했어", "좋아요", "좋아", "입니다", "이에요", "예요", "이요", "인데요", "인데", "과목", "교과목", "ai빅데이터학과", "빅데이터학과", "ai", "더", "많이", "계산하나요", "되어", "돼요", "인지",
}
# '얼마여야', '졸업하려면', '알려주실' 처럼 활용형으로 나타나는 일반 어휘의 어간
GENERIC_PREFIXES = ("얼마", "몇", "어떻", "무엇", "알려", "졸업하", "필요", "가능", "되나", "하나", "있나", "해야", "하려", "시작",
                     "도와", "도움", "있어", "안녕", "반가", "고마", "감사", "수고", "누구", "뭐", "뭘")
_PARTICLE_RE = re.compile(r"(하도록|도록|으려면|려면|는지|은지|으로는|에서는|에게는|인가요|입니까|하나요|한가요|나요|인지|이란|에서|으로|까지|부터|은|는|이|가|을|를|에|로|도|의|와|과|만|란)$")

_INJECTION = re.compile(
    r"(이전|위의|앞의|기존)\s*(지시|명령|규칙|프롬프트).{0,12}(무시|잊|따르지)|"
    r"(시스템|system)\s*(프롬프트|prompt|지시)|ignore\s+(all\s+)?(previous|prior|above)|"
    r"(프롬프트|지시문|instructions?).{0,10}(출력|공개|알려|보여|reveal|print)|jailbreak|developer\s*mode",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def extract_year(question: str) -> int | None:
    """질문에 명시된 학번. '2024학번' 또는 '24학번'을 인식합니다."""
    m = _YEAR_FULL.search(question or "")
    if m:
        return int(m.group(1))
    m = _YEAR_SHORT.search(question or "")
    if m and 10 <= int(m.group(1)) <= 40:
        return 2000 + int(m.group(1))
    return None


def is_injection(question: str) -> bool:
    return bool(_INJECTION.search(question or ""))


_WORDS: set[str] = set()
_FLAT = ""
_VERB_END = re.compile(r"(해야돼요|해야돼|해야해|해야지|하지|하면|하는|하나|하냐|할까|할래|할|한|했|되는|되나|될|된|도록|려면|하려|해야|해도|해서|하고|해요|돼요|돼|해)$")
_QUESTION_END = re.compile(r"(인가요|입니까|가요|나요|까요|니까)$")


def set_vocabulary(texts) -> None:
    """문서에 나오는 한글 낱말 집합을 등록합니다. 조사 판별에 사용합니다."""
    global _FLAT
    joined = unicodedata.normalize("NFC", " ".join(texts)).casefold()
    _WORDS.clear()
    _WORDS.update(re.findall(r"[가-힣]{2,}", joined))
    _FLAT = re.sub(r"\s+", "", joined)


def _strip_particle(term: str) -> str:
    """조사·어미를 뗍니다. 단, 문서의 실제 낱말인데 뗀 어간은 문서에 없으면 보존합니다(예: 비교과, 졸업및취창업지도)."""
    for _ in range(3):
        stripped = _VERB_END.sub("", _PARTICLE_RE.sub("", term))
        stripped = _QUESTION_END.sub("", stripped) if len(stripped) > 2 else stripped
        if stripped == term:
            break
        if len(stripped) < 2:
            return ""
        if term in _WORDS and stripped not in _WORDS:
            break
        term = stripped
    return term


def content_terms(question: str) -> list[str]:
    """질문에서 주제를 나타내는 핵심어만 남깁니다(조사·일반어·학번 숫자 제외)."""
    q = unicodedata.normalize("NFC", question or "").casefold()
    q = _YEAR_FULL.sub(" ", q)
    q = _YEAR_SHORT.sub(" ", q)
    semesters = [f"{a}학년{b}학기" for a, b in re.findall(r"(\d)\s*학년\s*(\d)\s*학기", q)]
    q = re.sub(r"\d\s*학년\s*\d\s*학기", " ", q)
    raw = semesters + re.findall(r"[가-힣]{2,}|[a-z][a-z0-9]{1,}|\d+\s*학점", q)
    terms: list[str] = []
    for token in raw:
        token = _strip_particle(token)
        if not token or token in GENERIC_TERMS or token.startswith(GENERIC_PREFIXES) or len(token) < 2 or re.fullmatch(r"\d+\s*학점", token):
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _bigrams(term: str) -> list[str]:
    return [term[i:i + 2] for i in range(len(term) - 1)] if len(term) >= 2 else [term]


def term_coverage(term: str, text: str) -> float:
    """핵심어가 본문에 얼마나 들어 있는지 0~1. 한국어 복합어('교양학점'→'교양'+'학점')를 2글자 단위로 비교합니다."""
    flat = re.sub(r"\s+", "", unicodedata.normalize("NFC", text).casefold())
    if term in flat:
        return 1.0
    if re.fullmatch(r"[a-z0-9]+", term) or re.search(r"\d", term):
        return 0.0  # 영문·숫자가 들어간 핵심어(예: 3학년1학기)는 부분 일치를 허용하지 않습니다.
    grams = _bigrams(term)
    return sum(1 for g in grams if g in flat) / len(grams)


# 문서는 '외국인'이라고 쓰는데 학생은 '유학생'이라고 묻는 식의 용어 차이(소규모 도메인 사전)
SYNONYMS = {"유학생": "외국인", "외국인학생": "외국인", "외국인유학생": "외국인"}


def normalize_query(text: str) -> str:
    """검색용 질의 보정: 도메인 동의어로 바꾸고, 문서에 없는 낱말은 첫 글자가 같고 아주 비슷한 문서 낱말로 고칩니다(오타 보정)."""
    import difflib
    if not _WORDS:
        return text

    def fix(match: "re.Match[str]") -> str:
        word = match.group(0)
        stem = _strip_particle(word)
        if not stem or len(stem) < 3:
            return word
        if stem in SYNONYMS:
            return word.replace(stem, SYNONYMS[stem], 1)
        if stem in _WORDS or stem in _FLAT or len(stem) < 4:
            return word
        close = [w for w in difflib.get_close_matches(stem, _WORDS, n=3, cutoff=0.75) if w[0] == stem[0]]
        return word.replace(stem, close[0], 1) if close else word

    return re.sub(r"[가-힣]{3,}", fix, text)


def has_domain_terms(question: str) -> bool:
    """질문의 핵심어 중 하나라도 문서 어휘에 있으면 졸업 자료와 관련된 질문으로 봅니다(인사·잡담·무관한 질문은 False)."""
    terms = content_terms(question)
    return any(term_coverage(t, _FLAT) >= 0.6 for t in terms) if _FLAT else bool(terms)


def evidence_coverage(question: str, texts: list[str], threshold: float = 0.6) -> tuple[float, list[str]]:
    """(핵심어 중 근거에서 확인된 비율, 확인되지 않은 핵심어 목록)."""
    terms = content_terms(question)
    if not terms:
        return 0.0, []
    joined = " ".join(texts)
    missing = [t for t in terms if term_coverage(t, joined) < threshold]
    return (len(terms) - len(missing)) / len(terms), missing


_WHEN = re.compile(r"언제|시기|기한|마감|일정|며칠|몇\s*월")
_WHEN_EVIDENCE = re.compile(r"\d+\s*월|\d+\s*일|학기|매주|매\s*학기|주차|기간|이후|이전|전까지|까지|후에|직전|마감")


def question_type_supported(question: str, texts: list[str]) -> bool:
    """'언제' 류 질문에는 시점 표현이 있는 근거가 있어야 합니다."""
    if _WHEN.search(question or ""):
        return any(_WHEN_EVIDENCE.search(t) for t in texts)
    return True


_GREETING = re.compile(r"^\s*(안녕|하이|헬로|hello|hi\b|반가|고마워|감사|수고)", re.IGNORECASE)


def is_greeting(question: str) -> bool:
    return bool(_GREETING.match(question or "")) and len(content_terms(question)) <= 1


def parse_credit_count(question: str) -> float | None:
    """'110학점 이수' 처럼 학생이 말한 이수 학점을 읽습니다. 0~300 범위만 인정합니다."""
    for m in re.finditer(r"(?<![\d~])(\d{1,3}(?:\.\d)?)\s*학점", question or ""):
        value = float(m.group(1))
        window = question[max(0, m.start() - 12): m.end() + 12]
        if re.search(r"이수|들었|들은|채웠|취득|수강|땄|따|있", window) and 0 <= value <= 300:
            return value
    return None
