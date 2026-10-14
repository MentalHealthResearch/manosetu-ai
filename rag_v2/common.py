from __future__ import annotations
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def path(key: str) -> Path:
    return (HERE / CFG[key]).resolve()


os.environ.setdefault("HF_HOME", str(path("hf_home")))


def load_qu_cache(mock: bool) -> dict:
    p = OUT / ("query_understanding_mock.json" if mock else "query_understanding.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_qu_cache(cache: dict, mock: bool) -> None:
    p = OUT / ("query_understanding_mock.json" if mock else "query_understanding.json")
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
