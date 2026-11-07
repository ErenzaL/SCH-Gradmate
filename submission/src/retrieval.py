from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .document_loader import PROJECT_ROOT, load_documents

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "rag_index"


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.casefold())
    # Korean substring tokens help BM25 match inflected/compound Korean queries.
    result = list(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 3:
            result.extend(word[i:i+2] for i in range(len(word)-1))
    return result



def _intent_anchors(query: str) -> list[str]:
    q = query.casefold()
    if "topcit" in q:
        return ["topcit"]
    if "외국인" in q or "유학생" in q:
        return ["외국인", "편입생", "계약학과"]
    if "봉사학습" in q and any(term in q for term in ("신청", "방법", "어떻게")):
        return ["자율봉사 신청", "봉사학습교육신청", "봉사활동 신청", "신청 및 등록"]
    if "전공필수" in q:
        return ["전공필수"]
    if ("비교과" in q or "다면적" in q) and any(term in q for term in ("졸업", "인증", "몇 점", "몇점")):
        return ["다면적 졸업인증", "점수소계", "졸업인증"]
    if any(term in q for term in ("졸업학점", "총 이수학점", "몇 학점")):
        return ["총 이수학점", "졸업학점", "졸업요건"]
    return []

def _corpus_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(MODEL.encode())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


def _load_or_build():
    import faiss
    records, warnings = load_documents()
    if not records:
        return records, None, None, warnings
    fingerprint = _corpus_hash(records)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "vectors.faiss"
    records_path = INDEX_DIR / "records.json"
    manifest_path = INDEX_DIR / "manifest.json"
    if index_path.exists() and records_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fingerprint and manifest.get("model") == MODEL:
                ix = faiss.read_index(str(index_path))
                saved = json.loads(records_path.read_text(encoding="utf-8"))
                if len(saved) == ix.ntotal == len(records):
                    bm = BM25Okapi([tokenize(r["text"]) for r in saved])
                    return saved, ix, bm, warnings
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    model = embedder()
    vectors = model.encode([r["text"] for r in records], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    ix = faiss.IndexFlatIP(vectors.shape[1]); ix.add(vectors)
    faiss.write_index(ix, str(index_path))
    records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "model": MODEL, "count": len(records)}, ensure_ascii=False, indent=2), encoding="utf-8")
    bm = BM25Okapi([tokenize(r["text"]) for r in records])
    return records, ix, bm, warnings


@lru_cache(maxsize=1)
def index():
    return _load_or_build()


def invalidate_index_cache() -> None:
    index.cache_clear()


def _year_matches(record: dict, year: int | None) -> bool:
    if year is None:
        return True
    years = record.get("admission_years")
    # Unlabeled/shared documents remain available to all cohorts.
    return years is None or int(year) in years


def search(query: str, year: int | None = None, limit: int = 5) -> tuple[list[dict], list[str]]:
    records, ix, bm, warnings = index()
    if not records or ix is None or bm is None:
        return [], warnings
    allowed = [i for i, record in enumerate(records) if _year_matches(record, year)]
    if not allowed:
        return [], warnings
    query_vector = embedder().encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    semantic_scores, semantic_ids = ix.search(query_vector, len(records))
    semantic_sim = {int(i): float(score) for i, score in zip(semantic_ids[0], semantic_scores[0]) if i >= 0}
    lexical_scores = bm.get_scores(tokenize(query))
    max_lexical = float(max(lexical_scores[i] for i in allowed)) if allowed else 0.0
    lexical_order = sorted(allowed, key=lambda i: lexical_scores[i], reverse=True)
    lex_rank = {int(i): rank for rank, i in enumerate(lexical_order, start=1)}
    # First remove passages with neither meaningful keyword evidence nor close semantic similarity.
    candidates = [i for i in allowed if lexical_scores[i] > 0 or semantic_sim.get(i, 0) >= 0.66]
    anchors = _intent_anchors(query)
    if anchors:
        focused = [i for i in candidates if any(anchor in records[i]["text"].casefold() for anchor in anchors)]
        candidates = focused
    if not candidates:
        return [], warnings
    fused = {}
    for i in candidates:
        lexical_norm = float(lexical_scores[i] / max_lexical) if max_lexical else 0.0
        semantic_norm = max(0.0, min(1.0, (semantic_sim.get(i, 0.0) - 0.30) / 0.45))
        fused[i] = 0.62 * lexical_norm + 0.38 * semantic_norm
    ranked = sorted(candidates, key=lambda i: fused[i], reverse=True)
    # Collapse near-identical excerpts from the same page/source.
    output, seen = [], set()
    for i in ranked:
        rec = records[i]
        normalized = re.sub(r"\W", "", rec["text"].casefold())[:220]
        key = (rec["source"], rec["page"], normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(rec | {"score": fused[i], "semantic_similarity": semantic_sim.get(i, 0.0), "keyword_score": float(lexical_scores[i]), "keyword_rank": lex_rank.get(i)})
        if len(output) >= limit:
            break
    return output, warnings


def dense_search(query: str, year: int | None = None, limit: int = 5) -> list[dict]:
    """FAISS 의미 검색 결과. 학번 미지정 자료는 공통 근거로 유지합니다."""
    records, ix, _, _ = index()
    if not records or ix is None:
        return []
    allowed = [i for i, record in enumerate(records) if _year_matches(record, year)]
    if not allowed:
        return []
    vector = embedder().encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, indices = ix.search(vector, len(records))
    output = []
    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if idx in allowed:
            output.append(records[idx] | {"score": float(score), "semantic_similarity": float(score)})
            if len(output) >= limit:
                break
    return output


def bm25_search(query: str, year: int | None = None, limit: int = 5) -> list[dict]:
    """BM25 키워드 검색 결과. 학번 필터를 동일하게 적용합니다."""
    records, _, bm, _ = index()
    if not records or bm is None:
        return []
    allowed = [i for i, record in enumerate(records) if _year_matches(record, year)]
    scores = bm.get_scores(tokenize(query))
    ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)
    return [records[i] | {"score": float(scores[i]), "keyword_score": float(scores[i])}
            for i in ranked[:limit] if scores[i] > 0]


def rerank_results(query: str, passages: list[dict], limit: int = 5) -> list[dict]:
    """하이브리드 후보를 한국어 질의어의 본문 포함도와 결합 점수로 재정렬합니다."""
    q = [token for token in tokenize(query) if len(token) >= 2]
    rescored = []
    for passage in passages:
        body = passage["text"].casefold()
        covered = sum(1 for token in set(q) if token in body) / max(1, len(set(q)))
        score = 0.7 * float(passage.get("score", 0.0)) + 0.3 * covered
        rescored.append(passage | {"rerank_score": score, "query_coverage": covered})
    return sorted(rescored, key=lambda item: item["rerank_score"], reverse=True)[:limit]
