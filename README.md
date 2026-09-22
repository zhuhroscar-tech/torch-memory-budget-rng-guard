[![Consolidated](https://img.shields.io/badge/consolidated-555555?style=flat)](https://github.com/zhuhroscar-tech/torch-correctness-guards)

# torch-memory-budget-rng-guard

This guard has moved into the consolidated package:

**https://github.com/zhuhroscar-tech/torch-correctness-guards**

Use the umbrella package instead:

```bash
python -m pip install 'torch-correctness-guards[torch]'
torch-guard run memory-budget-rng
```

Python API:

```python
from torch_correctness_guards import safe_compile
```

The original functionality is preserved as the `memory-budget-rng` guard, including the diagnostic for PyTorch issue [pytorch/pytorch#190758](https://github.com/pytorch/pytorch/issues/190758).

This repository is archived to reduce one-guard-per-repo fragmentation. It remains available for historical reference, but active maintenance continues in `torch-correctness-guards`.
