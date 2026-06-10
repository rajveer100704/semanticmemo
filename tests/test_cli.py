from __future__ import annotations

import sys
from pathlib import Path

import pytest

from semanticmemo import cli


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["SemanticMemo", *argv])
    cli.main()


def test_cli_stats_on_fresh_db(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(monkeypatch, "--db-path", str(tmp_path / "cache.db"), "stats")
    output = capsys.readouterr().out
    assert "entries=0" in output
    assert "total_hits=0" in output


def test_cli_export_feedback_with_no_feedback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_path = tmp_path / "feedback.jsonl"
    _run(
        monkeypatch,
        "--db-path",
        str(tmp_path / "cache.db"),
        "export-feedback",
        "--out",
        str(out_path),
    )
    output = capsys.readouterr().out
    assert "exported=0" in output


def test_cli_requires_a_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["SemanticMemo"])
    with pytest.raises(SystemExit):
        cli.main()
