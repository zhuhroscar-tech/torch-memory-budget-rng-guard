[![已合并](https://img.shields.io/badge/consolidated-555555?style=flat)](https://github.com/zhuhroscar-tech/torch-correctness-guards)

# torch-memory-budget-rng-guard

这个 guard 已迁移到统一维护的合并包：

**https://github.com/zhuhroscar-tech/torch-correctness-guards**

请改用 umbrella package：

```bash
python -m pip install 'torch-correctness-guards[torch]'
torch-guard run memory-budget-rng
```

Python API：

```python
from torch_correctness_guards import safe_compile
```

原有功能已保留为 `memory-budget-rng` guard，包括针对 PyTorch issue [pytorch/pytorch#190758](https://github.com/pytorch/pytorch/issues/190758) 的诊断。

本仓库已归档，以减少 one-guard-per-repo 的碎片化。历史内容仍可查看，后续维护在 `torch-correctness-guards` 中继续。
