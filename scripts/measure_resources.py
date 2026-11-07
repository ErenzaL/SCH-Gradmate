"""구성별 최대 메모리(RSS)와 지연시간을 깨끗한 하위 프로세스에서 측정합니다.

    python scripts/measure_resources.py

macOS 에서의 측정값이며 Render(Linux, 0.1 CPU)에서는 CPU 시간이 더 길어질 수 있습니다. 메모리는 대체로 비슷하거나 약간 큽니다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import os, sys, time, json, resource
sys.path.insert(0, os.getcwd())
for k in ("GEMINI_API_KEY","GOOGLE_API_KEY","OPENAI_API_KEY"): os.environ[k] = ""
def rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
out = {"after_import_mb": None}
t = time.time()
from src.chatbot import respond
from src import retrieval
out["after_import_mb"] = round(rss(), 1)
t = time.time(); retrieval.index(); out["index_load_s"] = round(time.time() - t, 2); out["after_index_mb"] = round(rss(), 1)
out["backend"] = retrieval.active_backend()
qs = ["졸업에 봉사활동이 필요한가요?", "봉사학습을 어떻게 신청하나요?", "외국인 유학생도 봉사학습을 이수해야 하나요?", "TOPCIT 수준별 점수는?", "2024학번 졸업학점은?"] * 4
respond(qs[0], 2024, use_llm=False)
lat = []
for q in qs:
    t = time.perf_counter(); respond(q, 2024, use_llm=False); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
out["median_query_ms"] = round(lat[len(lat) // 2], 1); out["p95_query_ms"] = round(lat[int(len(lat) * 0.95) - 1], 1)
out["peak_rss_mb"] = round(rss(), 1)
out["torch_loaded"] = "torch" in sys.modules
print(json.dumps(out))
'''


def run(label: str, backend: str) -> dict:
    import os
    env = {**os.environ, "RETRIEVAL_BACKEND": backend}
    proc = subprocess.run([sys.executable, "-I", "-c", CHILD], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if not line:
        return {"config": label, "error": proc.stderr[-300:]}
    return {"config": label, **json.loads(line[-1])}


def main() -> None:
    rows = [run("bm25 (Render 권장)", "bm25"), run("hybrid (BM25 + 의미 검색)", "hybrid")]
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "evaluation" / "results" / "resource_usage.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
