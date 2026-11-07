"""평가 라벨 보강: 답을 담은 문구가 실제로 들어 있는 (자료, 쪽)을 관련 자료에 추가합니다.

봉사학습 안내문 두 건(220818ver, 조교ver)처럼 같은 내용을 담은 자료가 있을 때 한쪽만 라벨에 있으면
정답 인용이 오답으로 계산됩니다. 이 스크립트는 문구가 모두 포함된 쪽만 추가하며, 기존 라벨은 지우지 않습니다.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.document_loader import load_documents  # noqa: E402

# 질문 ID -> 한 쪽의 본문에 모두 들어 있어야 하는 문구
PHRASES = {
    "Q17": ["1학점", "부여"], "Q18": ["외국인", "제외"], "Q19": ["이수조건"], "Q20": ["자율봉사", "봉사학습교육신청"],
    "Q21": ["1365", "인증서만"], "Q22": ["평가교육", "신청"], "Q23": ["휴학", "인정"], "Q24": ["헌혈", "4시간"],
    "Q25": ["졸업이 불가능"], "Q38": ["졸업이 불가능"], "Q53": ["8시간", "인정"], "Q54": ["36"],
}
SOURCES_ONLY = ("봉사활동 등록 가이드라인", "학과 봉사학습 지도")


def main() -> None:
    path = ROOT / "data" / "evaluation" / "questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    records, _ = load_documents()
    pages: dict[tuple[str, int], str] = {}
    for r in records:
        pages[(r["source"], r["page"])] = pages.get((r["source"], r["page"]), "") + "\n" + r["text"]
    for q in questions:
        phrases = PHRASES.get(q["question_id"])
        if not phrases:
            continue
        have = {(unicodedata.normalize("NFC", s["source"]), s["page"]) for s in q["relevant_sources"]}
        added = []
        for (source, page), text in pages.items():
            if not source.startswith(SOURCES_ONLY) or (source, page) in have:
                continue
            flat = text.replace(" ", "")
            if all(p.replace(" ", "") in flat for p in phrases):
                ids = [r["chunk_id"] for r in records if r["source"] == source and r["page"] == page]
                q["relevant_sources"].append({"source": source, "page": page, "chunk_ids": ids}); added.append((source[:12], page))
        # 모든 라벨의 chunk_ids 를 현재 청킹에 맞게 갱신
        if added:
            q["label_revision"] = (q.get("label_revision", "") + f" 2026-11-07: 문구 {phrases} 가 실제로 들어 있는 쪽 추가 {added}.").strip()
        print(q["question_id"], "added", added)
    for q in questions:
        for s in q["relevant_sources"]:
            s["chunk_ids"] = [r["chunk_id"] for r in records if r["source"] == unicodedata.normalize("NFC", s["source"]) and r["page"] == s["page"]
                              and (q["admission_year"] is None or r.get("admission_years") is None or q["admission_year"] in r["admission_years"])]
    path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
