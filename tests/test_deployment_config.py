"""배포 설정(Render/Streamlit)과 비밀 정보 위생 검사."""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ipynb", ".example", ".csv"}
KEY_PATTERNS = [re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), re.compile(r"sk-[A-Za-z0-9]{32,}"), re.compile(r"(?i)api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9_\-]{24,}['\"]")]


def test_render_blueprint_is_valid_and_safe():
    yaml = pytest.importorskip("yaml")
    svc = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))["services"][0]
    assert svc["type"] == "web" and svc["runtime"] == "python" and svc["plan"] == "free"
    assert "--server.address 0.0.0.0" in svc["startCommand"] and "$PORT" in svc["startCommand"]
    assert "app.py" in svc["startCommand"] and svc["healthCheckPath"] == "/_stcore/health"
    assert "requirements.txt" in svc["buildCommand"] and "build_index.py" in svc["buildCommand"]
    env = {e["key"]: e for e in svc["envVars"]}
    assert re.fullmatch(r"3\.\d+\.\d+", env["PYTHON_VERSION"]["value"])
    assert env["RETRIEVAL_BACKEND"]["value"] == "bm25"
    for secret in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        assert env[secret].get("sync") is False and "value" not in env[secret]


def test_streamlit_config_keeps_security_protections_and_is_headless():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["server"]["headless"] is True
    assert cfg["server"].get("enableCORS", True) is True and cfg["server"].get("enableXsrfProtection", True) is True
    assert cfg["server"]["maxUploadSize"] <= 10
    assert cfg["client"]["showErrorDetails"] == "none"


def test_runtime_requirements_are_pinned_and_exclude_heavy_or_dev_packages():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
    names = {l.split("==")[0].lower() for l in lines}
    assert names.isdisjoint({"torch", "sentence-transformers", "faiss-cpu", "jupyterlab", "pytest", "matplotlib", "openai", "ipykernel"})
    assert {"streamlit", "pymupdf", "rank-bm25", "pandas", "openpyxl", "google-genai"} <= names


def test_python_version_file_matches_blueprint():
    pytest.importorskip("yaml")
    assert (ROOT / ".python-version").read_text().strip().startswith("3.11")


def test_gitignore_excludes_secrets_caches_and_environments():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", ".venv/", "__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/", ".streamlit/secrets.toml", "data/processed/rag_index/*"):
        assert pattern in ignore, pattern
    assert "!.env.example" in ignore


def test_env_example_contains_placeholders_only():
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    assert values["GEMINI_API_KEY"] == "" and values["GEMINI_MODEL"] == ""
    assert "OPENAI_API_KEY" not in values or values["OPENAI_API_KEY"] == ""


def test_no_api_keys_are_committed_in_project_files():
    skip_dirs = {".venv", "__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints", "submission"}
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == ".env" or set(path.relative_to(ROOT).parts) & skip_dirs:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(p.search(text) for p in KEY_PATTERNS):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_app_never_prints_or_stores_credentials():
    for name in ("app.py", "src/rag.py", "src/chatbot.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"(print|st\.(write|markdown|text|code|caption))\([^)]*API_KEY", text), name

