"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed."""
from __future__ import annotations

import json

import pytest

import torch_memory_budget_rng_guard.core as core
from torch_memory_budget_rng_guard.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-memory-budget-rng-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert code in (0, 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_output_reports_error_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis and guarding"}
    assert code == 2


def test_torch_unavailable_text_output_shows_fail_headline_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch unavailable: torch is required for diagnosis and guarding" in out
    assert code == 2


def _fake_report(unguarded_bug, guard_ok):
    return {
        "torch_version": "0.0.0-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/190758",
        "unguarded_cases": [
            {
                "budget": 0.0,
                "x_grad": [[10.0]],
                "expected_grad_from_forward_mask": [[14.0]],
                "matches": not unguarded_bug,
            }
        ],
        "guarded_cases": [
            {
                "budget": 0.0,
                "x_grad": [[14.0]],
                "expected_grad_from_forward_mask": [[14.0]],
                "matches": guard_ok,
            }
        ],
        "any_unguarded_rng_recompute_bug": unguarded_bug,
        "guard_fully_correct": guard_ok,
    }


def test_no_bug_shows_info_message_not_fail(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(False, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[i] no unguarded RNG-recompute divergence reproduced" in out
    assert code == 0


def test_guard_failed_label_shown_when_guard_ineffective(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "GUARD-FAILED" in out
    assert "[X] safe_compile() did NOT prevent the divergence" in out
    assert code == 1


def test_bug_reproduced_and_guard_ok_shows_expected_headlines(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] unguarded torch.compile RNG-recompute gradient bug reproduced on this host (budget<1.0)" in out
    assert "[OK] safe_compile() prevents the divergence at every tested budget" in out
    assert code == 0
