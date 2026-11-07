"""배포 빌드 단계에서 실행하는 색인 산출물 생성 스크립트.

    python scripts/build_index.py            # 청크 JSON + 매니페스트 (Render Free 기본: 키워드 검색)
    python scripts/build_index.py --dense    # 선택: FAISS 벡터까지(sentence-transformers 필요)

Render 의 디스크는 재시작마다 초기화되므로 런타임 생성물에 의존하지 않고, 빌드마다 같은 입력에서 같은 산출물을 만듭니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import retrieval  # noqa: E402


def main() -> int:
    dense = "--dense" in sys.argv
    info = retrieval.build_artifacts(with_dense=dense)
    print(f"chunks={info['records']} dense={info['dense']} fingerprint={info['fingerprint'][:12]}")
    for warning in info["warnings"]:
        print("note:", warning)
    return 0 if info["records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
