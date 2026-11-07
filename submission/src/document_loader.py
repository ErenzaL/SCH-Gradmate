from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TEXT = {".pdf", ".txt", ".md", ".markdown"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CHUNK_SIZE = 1100
OVERLAP = 120
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


def _record(text: str, filename: str, page: int, category: str) -> dict:
    years = _years(text)
    return {
        "text": text.strip(), "source": filename, "page": page,
        "admission_year": years[0] if len(years) == 1 else None,
        "admission_years": years or None, "category": category,
    }


def _split_page(text: str, filename: str, page: int, category: str) -> list[dict]:
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
                chunks.append(_record(joined, filename, page, category))
            current = []

    for line in lines:
        years = _years(line)
        # A new year heading begins its own evidence chunk.
        is_heading = bool(years) and (len(line) < 100 or "점수소계" in line or "학번" in line)
        if is_heading and current and years != current_years:
            flush()
        if current and sum(map(len, current)) + len(line) + 1 > CHUNK_SIZE:
            tail = current[-2:] if OVERLAP else []
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


def _extract_pdf(path: Path, category: str) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    warnings: list[str] = []
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
                    table_records.extend(_table_records(table, path.name, page_number, category))
                records.extend(table_records)
                # For cohort tables, use structured rows to preserve year boundaries.
                # Other tables supplement, rather than replace, the page's surrounding text.
                if any(record.get("admission_years") for record in table_records):
                    text = ""
            image_count = len(page.get_images(full=True))
            if len(text) < 40:
                if not tables:
                    kind = "이미지 또는 스캔 페이지" if image_count or page.get_drawings() else "텍스트 추출 불가 페이지"
                    warnings.append(f"{path.name} {page_number}쪽: {kind}로 확인되어 문서 인식(OCR)이 필요합니다.")
                continue
            records.extend(_split_page(text, path.name, page_number, category))
    return records, warnings


def load_documents(directory: str | Path | None = None) -> tuple[list[dict], list[str]]:
    root = Path(directory) if directory is not None else PROJECT_ROOT / "data" / "documents"
    root = root.resolve()
    chunks: list[dict] = []
    warnings: list[str] = []
    if not root.exists():
        return [], [f"자료 폴더를 찾을 수 없습니다: {root}"]
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        filename = normalize_filename(path.name)
        suffix = path.suffix.lower()
        category = _category(filename)
        if suffix == ".pdf":
            docs, notes = _extract_pdf(path, category)
            chunks.extend(docs); warnings.extend(notes)
        elif suffix in {".txt", ".md", ".markdown"}:
            try:
                text = path.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp949").strip()
            if text:
                chunks.extend(_split_page(text, filename, 1, category))
        elif suffix in SUPPORTED_IMAGES:
            warnings.append(f"{filename}: 이미지 문서로 인식되어 문서 인식(OCR)이 필요합니다.")
    rules_path = root.parent / "graduation_rules.json"
    if rules_path.exists():
        import json
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        for cohort, rule in rules.items():
            if not isinstance(rule, dict) or "total_credits" not in rule:
                continue
            text = (f"{cohort}학번 졸업요건: 총 이수학점 {rule['total_credits']}학점. "
                    f"전공 {rule.get('major', {}).get('total', '미확인')}학점, "
                    f"교양 {rule.get('general_education', {}).get('total', '미확인')}학점. "
                    f"전공필수 {rule.get('major', {}).get('required', '미확인')}학점. "
                    f"다면적 졸업인증 기준 {rule.get('certification_points', '미확인')}점.")
            record = _record(text, rule.get("source", rules_path.name), rule.get("page", 1), "졸업요건")
            record["admission_year"] = int(cohort) if cohort.isdigit() else None
            record["admission_years"] = [int(cohort)] if cohort.isdigit() else None
            chunks.append(record)
    return chunks, warnings
