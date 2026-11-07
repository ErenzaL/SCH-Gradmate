from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TEXT = {".pdf", ".txt", ".md", ".markdown"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CHUNK_SIZE = 1100
OVERLAP = 120
TRANSCRIPTIONS_FILE = "image_transcriptions.json"
# 마지막 load_documents() 호출의 내부 처리 기록(중복 제거 등). 사용자 경고와 분리합니다.
LOAD_NOTES: list[str] = []
DEPARTMENT = "AI·빅데이터학과"
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})\s*학번")
RANGE_RE = re.compile(r"(?<!\d)(20\d{2})\s*(?:~|～|∼|[-–])\s*(20\d{2})(?!\d)")


def normalize_filename(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _category(filename: str) -> str:
    name = filename.lower()
    if any(word in name for word in ("졸업", "graduation", "자격요건")):
        return "졸업요건"
    if any(word in name for word in ("봉사", "volunteer")):
        return "봉사학습"
    if any(word in name for word in ("교육과정", "curriculum")):
        return "교육과정"
    return "학과자료"


def _years(text: str) -> list[int]:
    found: set[int] = set()
    for match in YEAR_RE.finditer(text):
        start = end = int(match.group(1))
        if 2000 <= start <= end <= 2099 and end - start <= 30:
            found.update(range(start, end + 1))
    for match in RANGE_RE.finditer(text):
        start, end = map(int, match.groups())
        if 2000 <= start <= end <= 2099 and end - start <= 30:
            found.update(range(start, end + 1))
    return sorted(found)


def _title(filename: str) -> str:
    return Path(filename).stem


def _department(filename: str) -> str | None:
    return DEPARTMENT if ("AI" in filename or "빅데이터" in filename) else None


def _filename_date(filename: str) -> str | None:
    """파일명에 적힌 날짜(예: _20240305, (220818ver))만 사용합니다. 없으면 None."""
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"\((\d{2})(\d{2})(\d{2})ver\)", filename)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _record(text: str, filename: str, page: int, category: str, section: str | None = None) -> dict:
    years = _years(text)
    return {
        "text": text.strip(), "source": filename, "page": page,
        "admission_year": years[0] if len(years) == 1 else None,
        "admission_years": years or None, "category": category,
        "title": _title(filename), "department": _department(filename),
        "document_date": _filename_date(filename), "section": section,
        "transcribed": False,
    }


STRUCT_HEADING_RE = re.compile(r"^(\[[^\]]{2,60}\]|\d{1,2}\.\s+\S)")
HEADING_RE = re.compile(r"^(\[[^\]]{2,60}\]|\d{1,2}\.\s*\S.{0,50}|※\s*\S.{0,50})")


def _split_page(text: str, filename: str, page: int, category: str,
                chunk_size: int | None = None, overlap: int | None = None) -> list[dict]:
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = OVERLAP if overlap is None else overlap
    section: str | None = None
    section_at_flush: str | None = None
    # Keep year-labeled table rows apart; splitting happens at paragraph/line boundaries.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    chunks: list[dict] = []
    current: list[str] = []
    current_years: list[int] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            joined = "\n".join(current).strip()
            if joined:
                chunks.append(_record(joined, filename, page, category, section_at_flush))
            current = []

    for line in lines:
        years = _years(line)
        if HEADING_RE.match(line):
            section = line[:60]
        if not current:
            section_at_flush = section
        # A new year heading begins its own evidence chunk.
        is_heading = bool(years) and (len(line) < 100 or "점수소계" in line or "학번" in line)
        if is_heading and current and years != current_years:
            flush()
        # FAQ 문항("13. ...")이나 [제목] 줄에서는 이미 내용이 있으면 새 청크를 시작합니다.
        if current and STRUCT_HEADING_RE.match(line) and sum(map(len, current)) > 120:
            flush()
        if current and sum(map(len, current)) + len(line) + 1 > chunk_size:
            tail = current[-2:] if overlap else []
            flush()
            current = tail
        current.append(line)
        current_years = years or current_years
    flush()
    return chunks


def _table_records(table, filename: str, page_number: int, category: str) -> list[dict]:
    rows = table.extract()
    has_cohort_rows = any((RANGE_RE.search(str(row[0] or "")) or re.fullmatch(r"\s*20\d{2}\s*", str(row[0] or "")) or ("학번" in str(row[0] or "") and _years(str(row[0])))) for row in rows if row)
    if not has_cohort_rows:
        serialized = [" | ".join(re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row) for row in rows if any(row)]
        return [_record("\n".join(serialized[i:i+18]), filename, page_number, category) for i in range(0, len(serialized), 18)]
    output: list[dict] = []
    current: list[str] = []
    cohort_years: list[int] | None = None
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if not any(cells):
            continue
        label = cells[0]
        explicit_years = _years(label)
        if re.fullmatch(r"\s*20\d{2}\s*", label):
            explicit_years = [int(label.strip())]
        range_match = RANGE_RE.search(label)
        if range_match:
            start, end = map(int, range_match.groups())
            if 0 <= end - start <= 30:
                explicit_years = list(range(start, end + 1))
        is_cohort_row = bool(explicit_years and ("학번" in label or range_match or re.fullmatch(r"\s*20\d{2}\s*", label)))
        if is_cohort_row:
            if current:
                record = _record("\n".join(current), filename, page_number, category)
                record["admission_years"] = cohort_years
                record["admission_year"] = cohort_years[0] if cohort_years and len(cohort_years) == 1 else None
                output.append(record)
            current = []
            cohort_years = explicit_years
        elif not current and len(output) == 0 and not explicit_years:
            # Discard generic table headings before a cohort-specific first row.
            continue
        current.append(" | ".join(cells))
    if current:
        record = _record("\n".join(current), filename, page_number, category)
        record["admission_years"] = cohort_years
        record["admission_year"] = cohort_years[0] if cohort_years and len(cohort_years) == 1 else None
        output.append(record)
    return output


def _extract_pdf(path: Path, category: str, chunk_size: int | None = None, overlap: int | None = None) -> tuple[list[dict], list[str]]:
    import pymupdf  # 런타임에 색인 산출물을 재사용하면 불러오지 않습니다.
    records: list[dict] = []
    warnings: list[str] = []
    stated_date: str | None = None
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, 1):
            text = page.get_text("text", sort=True).strip()
            try:
                tables = page.find_tables().tables
            except Exception:
                tables = []
            table_records = []
            if tables:
                for table in tables:
                    table_records.extend(_table_records(table, normalize_filename(path.name), page_number, category))
                records.extend(table_records)
                # For cohort tables, use structured rows to preserve year boundaries.
                # Other tables supplement, rather than replace, the page's surrounding text.
                if any(record.get("admission_years") for record in table_records):
                    text = ""
            if stated_date is None:
                m = re.search(r"\((20\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\s*기준\)", page.get_text("text"))
                if m:
                    stated_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            image_count = len(page.get_images(full=True))
            if len(text) < 40:
                if not tables:
                    kind = "이미지 또는 스캔 페이지" if image_count or page.get_drawings() else "텍스트 추출 불가 페이지"
                    warnings.append(f"{normalize_filename(path.name)} {page_number}쪽: {kind}로 확인되어 문서 인식(OCR)이 필요합니다.")
                continue
            records.extend(_split_page(text, normalize_filename(path.name), page_number, category, chunk_size, overlap))
    if stated_date:
        for record in records:
            record["document_date"] = record["document_date"] or stated_date
    return records, warnings


def _normalized_text_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).casefold()


def _finalize(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    """안정적인 chunk_id를 부여하고 같은 자료 안의 완전 중복 청크를 제거합니다.

    다른 자료와 본문이 같은 청크는 삭제하지 않고 `duplicate_of`에 먼저 나온 청크 ID를 기록합니다.
    """
    notes: list[str] = []
    counters: dict[tuple[str, int], int] = {}
    seen: dict[str, dict] = {}
    output: list[dict] = []
    for chunk in chunks:
        key = _normalized_text_key(chunk["text"])
        first = seen.get(key)
        if first is not None and first["source"] == chunk["source"]:
            notes.append(f"{chunk['source']} {chunk['page']}쪽: 같은 자료의 중복 청크를 제외했습니다.")
            continue
        slot = (chunk["source"], chunk["page"])
        counters[slot] = counters.get(slot, 0) + 1
        stem = hashlib.sha1(chunk["source"].encode("utf-8")).hexdigest()[:6]
        chunk["chunk_id"] = f"{stem}-p{chunk['page']}-{counters[slot]:02d}"
        chunk["duplicate_of"] = first["chunk_id"] if first is not None else None
        if first is None:
            seen[key] = chunk
        output.append(chunk)
    return output, notes


def _load_transcriptions(root: Path) -> list[dict]:
    path = root.parent / TRANSCRIPTIONS_FILE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in data.get("records", []):
        # 원본 이미지가 자료 폴더에 있을 때만 전사본을 사용합니다.
        source = normalize_filename(item["source"])
        if not any(normalize_filename(p.name) == source for p in root.iterdir() if p.is_file()):
            continue
        years = item.get("admission_years")
        records.append({
            "text": item["text"].strip(), "source": source, "page": item.get("page", 1),
            "admission_year": years[0] if years and len(years) == 1 else None,
            "admission_years": sorted(years) if years else None,
            "category": item.get("category", "학과자료"), "title": item.get("title", _title(source)),
            "department": item.get("department"), "document_date": item.get("document_date"),
            "section": item.get("section"), "transcribed": True,
        })
    return records


def load_documents(directory: str | Path | None = None, chunk_size: int | None = None,
                   overlap: int | None = None, include_transcriptions: bool = True) -> tuple[list[dict], list[str]]:
    root = Path(directory) if directory is not None else PROJECT_ROOT / "data" / "documents"
    root = root.resolve()
    chunks: list[dict] = []
    warnings: list[str] = []
    if not root.exists():
        return [], [f"자료 폴더를 찾을 수 없습니다: {root}"]
    transcribed = _load_transcriptions(root) if include_transcriptions else []
    transcribed_sources = {record["source"] for record in transcribed}
    rules_file = root.parent / "graduation_rules.json"
    if include_transcriptions and rules_file.exists():
        for rule in json.loads(rules_file.read_text(encoding="utf-8")).values():
            if isinstance(rule, dict) and "total_credits" in rule and rule.get("source"):
                transcribed_sources.add(normalize_filename(rule["source"]))
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        filename = normalize_filename(path.name)
        suffix = path.suffix.lower()
        category = _category(filename)
        if suffix == ".pdf":
            try:
                docs, notes = _extract_pdf(path, category, chunk_size, overlap)
            except Exception:
                warnings.append(f"{filename}: PDF를 읽지 못해 제외했습니다.")
                continue
            chunks.extend(docs)
            # 전사본이 있는 쪽은 OCR 필요 경고 대신 전사 사용 안내만 남깁니다.
            transcribed_pages = {r["page"] for r in transcribed if r["source"] == filename}
            for note in notes:
                m = re.search(r"(\d+)쪽", note)
                if m and int(m.group(1)) in transcribed_pages:
                    continue
                warnings.append(note)
        elif suffix in {".txt", ".md", ".markdown"}:
            try:
                text = path.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp949").strip()
            if text:
                chunks.extend(_split_page(text, filename, 1, category, chunk_size, overlap))
        elif suffix in SUPPORTED_IMAGES:
            if filename not in transcribed_sources:
                warnings.append(f"{filename}: 이미지 문서로 인식되어 문서 인식(OCR)이 필요합니다.")
    chunks.extend(transcribed)
    rules_path = root.parent / "graduation_rules.json"
    if rules_path.exists():
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        for cohort, rule in rules.items():
            if not isinstance(rule, dict) or "total_credits" not in rule:
                continue
            major, general = rule.get("major", {}), rule.get("general_education", {})
            text = (f"AI·빅데이터학과 {cohort}학번 졸업요건(교육과정 이수조건): 졸업학점(총 이수학점) {rule['total_credits']}학점. "
                    f"전공 {major.get('total', '미확인')}학점(전공기초 {major.get('basic', '미확인')}, 전공필수 {major.get('required', '미확인')}, "
                    f"전공선택 {major.get('elective', '미확인')}), "
                    f"교양 {general.get('total', '미확인')}학점(교양공통 {general.get('common', '미확인')}, 교양영역 {general.get('required_area', '미확인')}, "
                    f"교양선택 {general.get('elective', '미확인')}). "
                    f"전공필수 {major.get('required', '미확인')}학점. "
                    f"다면적 졸업인증 기준 {rule.get('certification_points', '미확인')}점.")
            record = _record(text, normalize_filename(rule.get("source", rules_path.name)), rule.get("page", 1), "졸업요건", "학번별 이수조건")
            record["admission_year"] = int(cohort) if cohort.isdigit() else None
            record["admission_years"] = [int(cohort)] if cohort.isdigit() else None
            record["department"] = DEPARTMENT
            record["transcribed"] = True
            chunks.append(record)
    chunks, dedupe_notes = _finalize(chunks)
    LOAD_NOTES[:] = dedupe_notes
    return chunks, warnings
