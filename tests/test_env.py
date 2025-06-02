import json
from pathlib import Path

import toml


def test_pyproject_toml_parses():
    p = Path(__file__).parent.parent / "pyproject.toml"
    cfg = toml.loads(p.read_text())
    assert "project" in cfg


def test_renovate_json_parses():
    p = Path(__file__).parent.parent / "renovate.json"
    cfg = json.loads(p.read_text())
    assert "extends" in cfg
