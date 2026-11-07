import re
import unicodedata

from src import document_loader as dl


def test_every_chunk_has_required_metadata_and_unique_stable_ids():
    records, _ = dl.load_documents()
    assert records
    ids = [r["chunk_id"] for r in records]
    assert len(ids) == len(set(ids))
    for r in records:
        for key in ("text", "source", "page", "title", "category", "admission_years", "department", "document_date", "section", "chunk_id", "transcribed"):
            assert key in r, key
        assert r["text"].strip() and r["page"] >= 1
        assert r["source"] == unicodedata.normalize("NFC", r["source"])
    again, _ = dl.load_documents()
    assert ids == [r["chunk_id"] for r in again]


def test_cohort_rows_keep_their_own_years():
    records, _ = dl.load_documents()
    dept = [r for r in records if r["source"] == "AI·빅데이터학과_20240305.pdf" and r["admission_years"]]
    by_years = {tuple(r["admission_years"]) for r in dept}
    assert (2023, 2024) in by_years and (2022,) in by_years and (2019, 2020) in by_years
    row_2022 = next(r for r in dept if r["admission_years"] == [2022])
    assert "95" in row_2022["text"] and "TOPCIT" not in row_2022["text"]


def test_transcribed_images_are_searchable_and_flagged():
    records, warnings = dl.load_documents()
    assert any(r["source"] == "졸업자격요건.png" and r["transcribed"] and "복수전공" in r["text"] for r in records)
    assert not any("졸업자격요건.png" in w for w in warnings)
    assert any("AI빅데이터학과 졸업여건.png" == r["source"] and r["admission_years"] == [2023, 2024, 2025, 2026] for r in records)


def test_unreadable_pdf_pages_are_reported_not_invented():
    _, warnings = dl.load_documents()
    assert any("6쪽" in w and "OCR" in w for w in warnings)


def test_document_dates_only_from_stated_sources():
    records, _ = dl.load_documents()
    dates = {r["source"]: r["document_date"] for r in records}
    assert dates["AI·빅데이터학과_20240305.pdf"] == "2024-03-05"
    assert dates["봉사활동 등록 가이드라인 및 자주묻는질문(220818ver).pdf"] == "2022-08-18"
    assert dates["다면적졸업인증제 비교과 배점 기준표.pdf"] == "2024-03-01"
    assert dates["졸업자격요건.png"] is None


def test_cross_source_duplicates_are_flagged_not_deleted():
    records, _ = dl.load_documents()
    flagged = [r for r in records if r["duplicate_of"]]
    assert flagged and all(r["duplicate_of"] in {x["chunk_id"] for x in records} for r in flagged)
    assert len({re.sub(r"\s", "", r["text"]) for r in records if r["source"] == records[0]["source"]}) == len([r for r in records if r["source"] == records[0]["source"]])


def test_chunk_size_parameters_change_chunking():
    small, _ = dl.load_documents(chunk_size=600, overlap=80)
    large, _ = dl.load_documents(chunk_size=1800, overlap=200)
    assert len(small) > len(large)


def test_missing_directory_and_corrupt_inputs_do_not_crash(tmp_path):
    records, warnings = dl.load_documents(tmp_path / "nope")
    assert records == [] and warnings
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "docs" / "note.txt").write_text("졸업요건 안내\n2024학번 130학점", encoding="utf-8")
    (tmp_path / "docs" / "legacy.txt").write_bytes("한글 cp949 문서".encode("cp949"))
    records, warnings = dl.load_documents(tmp_path / "docs")
    assert any("broken.pdf" in w for w in warnings)
    assert any("130학점" in r["text"] for r in records) and any("cp949" in r["text"] for r in records)


def test_decomposed_hangul_filenames_are_normalized(tmp_path):
    d = tmp_path / "docs"; d.mkdir()
    (d / unicodedata.normalize("NFD", "졸업요건.txt")).write_text("졸업학점 130", encoding="utf-8")
    records, _ = dl.load_documents(d)
    assert records[0]["source"] == unicodedata.normalize("NFC", "졸업요건.txt")


def test_warning_filenames_are_nfc_normalized():
    _, warnings = dl.load_documents()
    assert warnings and all(w == unicodedata.normalize("NFC", w) for w in warnings)
