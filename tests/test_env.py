"""Smoke test for environment, dependencies, and configuration setup.
"""

from pathlib import Path
import pytest
import yaml

import finreason


def test_package_metadata():
    """Verify package import and version."""
    assert finreason.__version__ == "0.1.0"


def test_config_loads_properly():
    """Verify default configuration file is valid YAML and contains required keys."""
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs" / "default.yaml"
    assert config_path.exists(), f"Config file missing at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "system" in config
    assert "llm" in config
    assert "sandbox" in config
    assert "memory" in config
    assert "evals" in config

    # Check cognitive tiering config keys
    assert "router" in config["llm"]
    assert "tier1_fast" in config["llm"]
    assert "tier2_standard" in config["llm"]
    assert "tier3_forensic" in config["llm"]

    # Check sandbox quota keys
    assert config["sandbox"]["engine"] == "duckdb"
    assert config["sandbox"]["timeout_seconds"] > 0
    assert config["evals"]["absolute_tolerance_pct"] == 0.001


def test_core_dependencies_available():
    """Verify critical deterministic execution engines and libraries."""
    import duckdb
    import pandas as pd
    import pydantic
    import instructor

    # Quick smoke test of duckdb in-memory query
    con = duckdb.connect(database=":memory:")
    res = con.execute("SELECT 42 AS answer").fetchone()
    assert res[0] == 42
    con.close()
