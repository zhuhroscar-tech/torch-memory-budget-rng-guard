"""Command-line interface: run the from-scratch diagnosis of the
torch.compile activation_memory_budget RNG-recompute gradient
corruption bug (pytorch/pytorch#190758) against the currently installed
torch build, and confirm safe_compile() prevents it, using the shared
semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-memory-budget-rng-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile AOTAutograd partitioner silently corrupts "
            "gradients when activation_memory_budget < 1.0 and the "
            "compiled graph contains an RNG op (pytorch/pytorch#190758: "
            "the partitioner-forced recompute redraws a FRESH dropout "
            "mask in backward instead of replaying the forward's own "
            "draw) -- and verify safe_compile() prevents the divergence "
            "by forcing budget=1.0 for the duration of the guarded call. "
            "Never trusts a cached or previously-reported result, always "
            "re-runs the repro on THIS host's actual installed torch "
            "version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-memory-budget-rng-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_unguarded_rng_recompute_bug"]:
        print(status_headline(style, "fail", "unguarded torch.compile RNG-recompute gradient bug reproduced on this host (budget<1.0)"))
    else:
        print(status_headline(style, "info", "no unguarded RNG-recompute divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compile() prevents the divergence at every tested budget, and restores the caller's budget afterward"))
    else:
        print(status_headline(style, "fail", "safe_compile() did NOT prevent the divergence at at least one tested budget"))

    section("unguarded cases (budget -> actual grad vs grad implied by forward's own mask)")
    for c in report["unguarded_cases"]:
        flag = "ok" if c["matches"] else "WRONG-GRADIENT"
        print_fields([(f"budget={c['budget']}", flag)])

    section("guarded cases (safe_compile forces budget=1.0 internally, regardless of caller's requested budget)")
    for c in report["guarded_cases"]:
        flag = "guard-correct" if c["matches"] else "GUARD-FAILED"
        print_fields([(f"requested_budget={c['budget']}", flag)])

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
