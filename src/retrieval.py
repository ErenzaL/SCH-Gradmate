from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import document_loader
from .query_utils import GENERIC_PREFIXES, GENERIC_TERMS, _strip_particle, set_vocabulary
from .document_loader import PROJECT_ROOT, load_documents

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "rag_index"
ARTIFACT_VERSION = 3
# 색인용 텍스트 앞에 자료명·학과·학번·절 제목을 붙입니다(표시용 text 는 그대로). 실험으로 효과를 비교합니다.
CONTEXTUAL_HEADERS = os.getenv("RETRIEVAL_HEADERS", "0") == "1"  # 실험 결과 이득이 불분명해 기본은 끔
METHODS = ("bm25", "dense", "hybrid", "rrf")
RRF_K = 60


def index_text(record: dict, contextual: bool | None = None) -> str:
    """BM25/임베딩에 넣는 텍스트. 문맥 헤더(자료명·학과·학번·절)를 선택적으로 앞에 붙입니다."""
    if not (CONTEXTUAL_HEADERS if contextual is None else contextual):
        return record["text"]
    years = record.get("admission_years")
    year_part = ""
    if years:
        year_part = f"{years[0]}~{years[-1]}학번" if len(years) > 1 else f"{years[0]}학번"
    header = " ".join(x for x in (record.get("title"), record.get("department"), year_part, record.get("section")) if x)
    return f"{header}\n{record['text']}" if header else record["text"]


# ---------------------------------------------------------------- tokenization
_SEMESTER_RE = re.compile(r"(\d)\s*학년\s*(\d)\s*학기")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    # '3학년 1학기' 같은 학기 표기는 숫자가 흔해 구분이 안 되므로 하나의 토큰으로도 추가합니다.
    semester_tokens = [f"{a}학년{b}학기" for a, b in _SEMESTER_RE.findall(text)]
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text)
    # Korean substring tokens help BM25 match inflected/compound Korean queries.
    result = list(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 3:
            result.extend(word[i:i+2] for i in range(len(word)-1))
    return result + semester_tokens


def query_tokens(query: str, drop_generic: bool = True) -> list[str]:
    """BM25 질의 토큰. '어떻게', '필요한가요' 같은 일반 어휘는 점수를 흐리므로 제외합니다(드문 단어가 더 크게 반영됨)."""
    tokens = tokenize(query)
    q = unicodedata.normalize("NFC", query or "").casefold()
    # 조사가 붙은 질의 낱말('봉사학습을')이 문서의 낱말('봉사학습')과 맞도록 어간 토큰을 추가합니다.
    for word in re.findall(r"[가-힣]{2,}", q):
        stem = _strip_particle(word)
        if stem and stem != word:
            tokens.append(stem)
    if not drop_generic:
        return tokens
    generic_words = set()
    for word in re.findall(r"[가-힣]{2,}", q):
        stem = _strip_particle(word)
        if not stem or stem in GENERIC_TERMS or stem.startswith(GENERIC_PREFIXES):
            generic_words.add(word)
    if not generic_words:
        return tokens
    blocked = set(generic_words)
    for word in generic_words:
        if len(word) >= 3:
            blocked.update(word[i:i + 2] for i in range(len(word) - 1))
    # 일반 어휘와 같은 이음절 토큰이 핵심어에서도 나온 경우를 지키기 위해, 핵심어가 만드는 토큰은 남깁니다.
    keep = set()
    for word in re.findall(r"[가-힣]{2,}", q):
        if word not in generic_words:
            keep.add(word)
            if len(word) >= 3:
                keep.update(word[i:i + 2] for i in range(len(word) - 1))
    return [t for t in tokens if t not in blocked or t in keep]


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
    if ("비교과" in q or "다면적" in q) and any(term in q for term in ("졸업", "인증")):
        return ["다면적 졸업인증", "점수소계", "졸업인증"]
    if any(term in q for term in ("졸업학점", "총 이수학점", "몇 학점")):
        return ["총 이수학점", "졸업학점", "졸업요건"]
    return []


# ---------------------------------------------------------------- backend config
def requested_backend() -> str:
    value = (os.getenv("RETRIEVAL_BACKEND") or "auto").strip().lower()
    return value if value in {"auto", "bm25", "hybrid"} else "auto"


@lru_cache(maxsize=1)
def dense_available() -> bool:
    """의미 검색(sentence-transformers + faiss)을 이 환경에서 쓸 수 있는지 확인합니다."""
    if requested_backend() == "bm25":
        return False
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


# ---------------------------------------------------------------- build artifact
def source_fingerprint() -> str:
    """자료 원본 + 전사본 + 규칙 JSON + 청킹 설정의 지문. 내용이 바뀌면 산출물을 재생성합니다."""
    digest = hashlib.sha256()
    root = PROJECT_ROOT / "data"
    docs = root / "documents"
    for path in sorted(docs.iterdir(), key=lambda p: unicodedata.normalize("NFC", p.name)) if docs.exists() else []:
        if path.is_file() and not path.name.startswith("."):
            digest.update(unicodedata.normalize("NFC", path.name).encode("utf-8"))
            digest.update(path.read_bytes())
    for name in (document_loader.TRANSCRIPTIONS_FILE, "graduation_rules.json"):
        f = root / name
        if f.exists():
            digest.update(name.encode()); digest.update(f.read_bytes())
    digest.update(f"{document_loader.CHUNK_SIZE}:{document_loader.OVERLAP}:{ARTIFACT_VERSION}".encode())
    # 청킹 로직이 바뀌면 산출물도 다시 만들도록 로더 소스 자체를 지문에 포함합니다.
    digest.update(Path(document_loader.__file__).read_bytes())
    return digest.hexdigest()


def _paths(index_dir: Path | None = None) -> tuple[Path, Path, Path]:
    d = index_dir or INDEX_DIR
    return d / "records.json", d / "manifest.json", d / "vectors.faiss"


def _write_records(records_path: Path, manifest_path: Path, fingerprint: str, records: list[dict], warnings: list[str]) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps({"records": records, "warnings": warnings}, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "count": len(records), "artifact_version": ARTIFACT_VERSION,
                                         "model": None, "dense": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_artifacts(with_dense: bool | None = None, index_dir: Path | None = None) -> dict:
    """빌드 단계에서 호출합니다. 청크 JSON(항상)과 FAISS 벡터(선택)를 data/processed에 저장합니다."""
    records_path, manifest_path, vectors_path = _paths(index_dir)
    records, warnings = load_documents()
    fingerprint = source_fingerprint()
    _write_records(records_path, manifest_path, fingerprint, records, warnings)
    dense = False
    if with_dense is None:
        with_dense = dense_available()
    if with_dense and records:
        import faiss
        vectors = embedder().encode([index_text(r) for r in records], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        ix = faiss.IndexFlatIP(vectors.shape[1]); ix.add(vectors)
        faiss.write_index(ix, str(vectors_path))
        dense = True
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"model": MODEL, "dense": True, "dimension": int(vectors.shape[1])})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    elif vectors_path.exists():
        vectors_path.unlink()
    return {"records": len(records), "dense": dense, "fingerprint": fingerprint, "warnings": warnings}


def _load_records(index_dir: Path | None = None) -> tuple[list[dict], list[str], bool]:
    """(records, warnings, from_artifact). 산출물 지문이 현재 자료와 같을 때만 재사용합니다."""
    records_path, manifest_path, _ = _paths(index_dir)
    try:
        if records_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == source_fingerprint() and manifest.get("artifact_version") == ARTIFACT_VERSION:
                saved = json.loads(records_path.read_text(encoding="utf-8"))
                if isinstance(saved, dict) and len(saved.get("records", [])) == manifest.get("count"):
                    return saved["records"], list(saved.get("warnings", [])), True
    except (ValueError, OSError, json.JSONDecodeError):
        pass
    records, warnings = load_documents()
    return records, warnings, False


def _load_dense(records: list[dict], index_dir: Path | None = None):
    import faiss
    _, manifest_path, vectors_path = _paths(index_dir)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fresh = manifest.get("fingerprint") == source_fingerprint() and manifest.get("artifact_version") == ARTIFACT_VERSION
        if fresh and manifest.get("dense") and manifest.get("model") == MODEL and vectors_path.exists():
            ix = faiss.read_index(str(vectors_path))
            if ix.ntotal == len(records):
                return ix
    except (ValueError, OSError, json.JSONDecodeError):
        pass
    vectors = embedder().encode([index_text(r) for r in records], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    ix = faiss.IndexFlatIP(vectors.shape[1]); ix.add(vectors)
    return ix


def _load_or_build():
    records, warnings, _ = _load_records()
    warnings = list(warnings)
    if not records:
        return records, None, None, warnings
    bm = BM25Okapi([tokenize(index_text(r)) for r in records])
    set_vocabulary(r["text"] for r in records)
    ix = None
    if dense_available():
        try:
            ix = _load_dense(records)
        except Exception:
            ix = None
            warnings.append("의미 검색 모델을 불러오지 못해 키워드 검색만 사용합니다.")
    return records, ix, bm, warnings


@lru_cache(maxsize=1)
def index():
    return _load_or_build()


def invalidate_index_cache() -> None:
    index.cache_clear()


def active_backend() -> str:
    _, ix, _, _ = index()
    return "hybrid" if ix is not None else "bm25"


# ---------------------------------------------------------------- retrieval
def _year_matches(record: dict, year: int | None) -> bool:
    if year is None:
        return True
    years = record.get("admission_years")
    # Unlabeled/shared documents remain available to all cohorts.
    return years is None or int(year) in years


def _dense_scores(query: str, ix, n: int) -> dict[int, float]:
    vector = embedder().encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, ids = ix.search(vector, n)
    return {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}


def retrieve(query: str, year: int | None = None, limit: int = 5, method: str = "hybrid", rerank: bool = False,
             use_anchors: bool | str = True, collapse_duplicates: bool = False, corpus=None, drop_generic: bool = True) -> list[dict]:
    """공통 검색 진입점. method: bm25 | dense | hybrid | rrf.

    corpus=(records, ix, bm) 를 주면 그 말뭉치로 검색합니다(청크 크기 실험용).
    dense/hybrid/rrf 는 의미 색인이 없으면 bm25 로 대체되며 결과의 `method_used` 에 표시됩니다.
    """
    if corpus is None:
        records, ix, bm, _ = index()
    else:
        records, ix, bm = corpus
    if not records or bm is None or not (query or "").strip():
        return []
    method_used = method if (method == "bm25" or ix is not None) else "bm25"
    allowed = [i for i, record in enumerate(records) if _year_matches(record, year)]
    if not allowed:
        return []
    lexical = bm.get_scores(query_tokens(query, drop_generic))
    lex_order = sorted(allowed, key=lambda i: lexical[i], reverse=True)
    lex_rank = {i: r for r, i in enumerate(lex_order, start=1)}
    dense = _dense_scores(query, ix, len(records)) if method_used != "bm25" else {}
    dense_order = sorted(allowed, key=lambda i: dense.get(i, -1.0), reverse=True) if dense else []
    dense_rank = {i: r for r, i in enumerate(dense_order, start=1)}

    if method_used == "bm25":
        candidates = [i for i in allowed if lexical[i] > 0]
        fused = {i: float(lexical[i]) for i in candidates}
    elif method_used == "dense":
        candidates = list(allowed)
        fused = {i: dense.get(i, 0.0) for i in candidates}
    elif method_used == "rrf":
        candidates = [i for i in allowed if lexical[i] > 0 or dense.get(i, 0) >= 0.66]
        fused = {i: 1.0 / (RRF_K + lex_rank[i]) + 1.0 / (RRF_K + dense_rank[i]) for i in candidates}
    else:  # hybrid: linear fusion of normalized BM25 and cosine similarity
        max_lex = float(max(lexical[i] for i in allowed)) or 0.0
        candidates = [i for i in allowed if lexical[i] > 0 or dense.get(i, 0) >= 0.66]
        fused = {}
        for i in candidates:
            lex_norm = float(lexical[i] / max_lex) if max_lex else 0.0
            sem_norm = max(0.0, min(1.0, (dense.get(i, 0.0) - 0.30) / 0.45))
            fused[i] = 0.62 * lex_norm + 0.38 * sem_norm
    anchors = _intent_anchors(query) if use_anchors else []
    if anchors and use_anchors == "promote":
        # 의도 앵커 문구가 든 청크를 앞으로 올리되(원래 점수순 유지) 나머지도 뒤에 남깁니다.
        hit_set = {i for i in candidates if any(a in records[i]["text"].casefold() for a in anchors)}
        ranked = sorted(candidates, key=lambda i: (i in hit_set, fused[i]), reverse=True)
    else:
        if anchors:  # 과거 동작(use_anchors=True): 앵커가 없는 청크는 제외
            candidates = [i for i in candidates if any(a in records[i]["text"].casefold() for a in anchors)]
        ranked = sorted(candidates, key=lambda i: fused[i], reverse=True)
    pool = limit * 3 if rerank else limit * 2
    output, seen_text, seen_dup = [], set(), set()
    for i in ranked:
        rec = records[i]
        key = (rec["source"], rec["page"], re.sub(r"\W", "", rec["text"].casefold())[:220])
        if key in seen_text:
            continue
        seen_text.add(key)
        if collapse_duplicates:
            origin = rec.get("duplicate_of") or rec.get("chunk_id")
            if origin in seen_dup:
                continue
            seen_dup.add(origin)
        output.append(rec | {"score": float(fused[i]), "semantic_similarity": dense.get(i), "keyword_score": float(lexical[i]),
                             "keyword_rank": lex_rank.get(i), "method_used": method_used})
        if len(output) >= pool:
            break
    if rerank:
        output = rerank_results(query, output, limit)
    return output[:limit]


def search(query: str, year: int | None = None, limit: int = 5) -> tuple[list[dict], list[str]]:
    """기본(하이브리드) 검색. 기존 호출부와 호환됩니다."""
    _, _, _, warnings = index()
    # 의미 색인이 없으면 retrieve() 가 BM25 로 대체합니다. 의도 앵커는 실험에서 이득이 없어 기본으로 끕니다.
    return retrieve(query, year, limit, method="hybrid", use_anchors="promote", collapse_duplicates=True), warnings


def dense_search(query: str, year: int | None = None, limit: int = 5) -> list[dict]:
    """FAISS 의미 검색 결과. 학번 미지정 자료는 공통 근거로 유지합니다."""
    return retrieve(query, year, limit, method="dense", use_anchors=False)


def bm25_search(query: str, year: int | None = None, limit: int = 5) -> list[dict]:
    """BM25 키워드 검색 결과. 학번 필터를 동일하게 적용합니다."""
    return retrieve(query, year, limit, method="bm25", use_anchors=False)


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


def build_corpus(records: list[dict], with_dense: bool = True, contextual: bool | None = None):
    """임의의 청크 목록으로 메모리 내 색인을 만듭니다(실험용, 디스크에 쓰지 않음)."""
    bm = BM25Okapi([tokenize(index_text(r, contextual)) for r in records])
    ix = None
    if with_dense and dense_available():
        import faiss
        vectors = embedder().encode([index_text(r, contextual) for r in records], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        ix = faiss.IndexFlatIP(vectors.shape[1]); ix.add(vectors)
    return records, ix, bm
