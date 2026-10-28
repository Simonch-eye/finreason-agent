"""
tests/test_cli.py
=================
Tests for finreason command line interface (CLI).
"""

import sys
import pytest
from finreason.cli import main


def test_cli_version(capsys):
    """Verify that version subcommand outputs version information."""
    sys.argv = ["finreason", "version"]
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FinReason-Agent:" in captured.out
    assert "Architecture:" in captured.out


def test_cli_eval_suite_20(capsys):
    """Verify that eval subcommand runs seed benchmark."""
    sys.argv = ["finreason", "eval", "--suite", "20"]
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "双轨评测基准大屏" in captured.out or "评测考场" in captured.out


def test_cli_audit_command(capsys, tmp_path):
    """Verify that audit subcommand executes and outputs memorandum."""
    out_file = tmp_path / "audit_memo.md"
    sys.argv = [
        "finreason", "audit",
        "-q", "核算公司2023年流动比率与资产负债率",
        "-c", "测试企业A",
        "-p", "2023年度",
        "-o", str(out_file),
    ]
    exit_code = main()
    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "金融刑侦自主审计意见书" in captured.out
    assert "测试企业A" in captured.out


def test_cli_export_data(capsys, tmp_path):
    """Verify that export-data subcommand generates finetuning datasets."""
    export_dir = tmp_path / "finetune_trajectories"
    sys.argv = ["finreason", "export-data", "-o", str(export_dir)]
    exit_code = main()
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "数据集导出完成" in captured.out
    assert (export_dir / "alpaca_sft.json").exists()
    assert (export_dir / "dpo_pairs.json").exists()


def test_cli_help(capsys):
    """Verify that --help lists all subcommands including ui."""
    sys.argv = ["finreason", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "audit" in captured.out
    assert "eval" in captured.out
    assert "ui" in captured.out
    assert "serve" in captured.out

