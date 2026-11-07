import os
import sys
from pathlib import Path

# 테스트는 무거운 임베딩 모델 없이 키워드(BM25) 검색으로 실행합니다. src 를 불러오기 전에 설정해야 합니다.
os.environ["RETRIEVAL_BACKEND"] = "bm25"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(autouse=True)
def no_external_ai(monkeypatch):
    """테스트가 실제 AI API를 호출하거나 .env 의 키를 읽지 않도록 모든 자격 증명을 비웁니다."""
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "")
    import src.rag as rag
    monkeypatch.setattr(rag, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(rag, "_llm_blocked_until", 0.0)  # 서킷 브레이커 상태가 테스트 사이에 새지 않게 초기화
    yield
