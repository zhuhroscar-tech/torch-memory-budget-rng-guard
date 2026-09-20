[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-memory-budget-rng-guard

Guards a real PyTorch correctness bug: [pytorch/pytorch#190758](https://github.com/pytorch/pytorch/issues/190758)
(**open** at time of writing). When `torch.compile` is used with
`torch._functorch.config.activation_memory_budget < 1.0` and the
compiled graph contains an RNG op (`F.dropout`, `randn_like`, ...), the
AOTAutograd partitioner can force that op to be **recomputed in the
backward graph** — and that recompute draws **fresh randomness**
instead of replaying the forward pass's own draw. The backward then
applies a different dropout mask than the forward actually used,
silently corrupting gradients. **No error, no warning.**

Root cause (per the upstream issue's own trace): the `budget == 0`
shortcut skips the knapsack partitioner's "random op" recompute ban
entirely, and at `0 < budget < 1` the knapsack path can still re-allow
recompute of banned RNG ops because `get_recomputable_banned_nodes` has
no RNG filter. The bug does **not** reproduce at `budget == 1.0` (the
default — no forced recompute at all).

**Training-level impact**, per the upstream issue's own reported
ablation (small two-dropout MLP, synthetic classification, 5 seeds,
CPU): eager mean eval accuracy 0.894 vs compiled at `budget=0` on
current main collapsing to **0.55–0.61 (near chance) with NaN final
train loss**. This is a real training-correctness failure, not a
cosmetic numeric mismatch.

Independently reproduced on this host (torch 2.14.0, CPU, Inductor
backend) using the issue author's own minimal reproducer: comparing
the actual backward gradient against the gradient *implied* by the
mask the compiled forward itself applied (inferred from which output
elements are exactly zero) — a mismatch proves the backward redrew a
different mask than the forward used, without needing internal
PyTorch instrumentation.

## What this tool can and cannot do

This is a **userspace call-boundary mitigation**, not a fix to
PyTorch's partitioner — no third-party package can patch AOTAutograd's
internal recompute-node selection from the outside. `safe_compile()`
temporarily forces `activation_memory_budget = 1.0` (the one value
confirmed **not** to trigger the bug) for the duration of a guarded
compiled call, then restores whatever budget the caller had configured
— even if the call raises. This sacrifices `budget<1.0`'s memory
savings for that call in exchange for gradient correctness: an
explicit, documented trade-off, never a silent one.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build. From source:

```bash
git clone https://github.com/zhuhroscar-tech/torch-memory-budget-rng-guard.git
cd torch-memory-budget-rng-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-memory-budget-rng-guard
torch-memory-budget-rng-guard --json
```

If you already manage a compatible PyTorch installation, install `.`
without the extra. Use `--no-color` for plain text output.

Exit codes: **0** means `safe_compile()` prevented the divergence at
every tested budget (and prints an "info" line, not a warning, if the
underlying unguarded bug also happened not to reproduce), **1** means
the guard failed to prevent the divergence at at least one tested
budget, and **2** means PyTorch could not be imported.

## Python API

Wrap any `torch.compile`d call site whose graph may contain an RNG op
and whose caller has `activation_memory_budget < 1.0` configured:

```python
from torch_memory_budget_rng_guard import safe_compile

# instead of:
#   compiled = torch.compile(model_fn)          # silently wrong gradients
#   out = compiled(x)                            # if budget<1.0 and an
#                                                  # RNG op gets recomputed
guarded = safe_compile(model_fn)
out = guarded(x)   # forces budget=1.0 for this call, then restores yours
```

Callers who need `budget<1.0`'s memory savings AND have independently
verified (e.g. via `diagnose()` against their own installed torch
build) that their specific graph contains no RNG ops may bypass this
guard and call `torch.compile` directly with their chosen budget.

## Scope and limitations

- Reproduced and tested on CPU only (torch 2.14.0). The upstream issue
  also reports the bug on CUDA builds; not independently verified on
  CUDA/MPS by this tool.
- The mitigation is coarse: it forces budget to exactly 1.0 for the
  *entire* guarded call, not a finer per-node RNG-aware budget. If your
  graph has no RNG ops at all, this guard has no effect other than the
  overhead of the context-manager wrapper — you can safely skip it.
- Does not detect or guard `randn_like`/other RNG ops used *outside* a
  `torch.compile`d region, or graphs compiled directly by the caller
  without going through `safe_compile()`.
- `diagnose()` always re-runs the actual reproduction against whatever
  torch build is installed — it never assumes a specific PyTorch
  version is or isn't affected. If the upstream issue is fixed and
  merged into a stable release, `any_unguarded_rng_recompute_bug` will
  correctly report `False` and this tool becomes a no-op safety net
  rather than a required workaround.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_memory_budget_rng_guard --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).
