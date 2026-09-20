[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-memory-budget-rng-guard

守护一个真实存在的 PyTorch 正确性缺陷：[pytorch/pytorch#190758](https://github.com/pytorch/pytorch/issues/190758)
（截至撰写本文仍为 **open** 状态）。当 `torch.compile` 与
`torch._functorch.config.activation_memory_budget < 1.0` 一起使用，
且被编译的计算图中包含随机数算子（如 `F.dropout`、`randn_like` 等）时，
AOTAutograd 的分区器可能强制该算子在**反向图中重新计算** —— 而这次重新
计算会抽取**全新的随机数**，而不是重放前向传播时已经使用过的那次抽取。
于是反向传播应用的 dropout 掩码与前向传播实际使用的掩码不同，导致梯度
被静默污染。**没有任何报错，没有任何警告。**

根因（依据上游 issue 自身的追踪结果）：`budget == 0` 的快捷路径完全跳过
了 knapsack 分区器对"随机算子"重新计算的禁令；而在 `0 < budget < 1` 时，
knapsack 路径仍可能重新允许被禁止的随机算子重新计算，因为
`get_recomputable_banned_nodes` 没有针对随机算子的过滤逻辑。该缺陷在
`budget == 1.0`（默认值，完全不强制重新计算）时**不会**复现。

**训练层面的影响**，依据上游 issue 自身报告的消融实验（一个含两个
dropout 层的小型 MLP，合成分类任务，5 个随机种子，CPU）：eager 模式的
平均评估准确率为 0.894，而在当前 main 分支、`budget=0` 下编译运行则
**崩溃至 0.55–0.61（接近随机猜测），且最终训练损失为 NaN**。这是真实
的训练正确性故障，而不是无关紧要的数值误差。

本仓库在本机(torch 2.14.0，CPU，Inductor 后端)上使用 issue 作者自己
给出的最小复现代码独立复现了该问题：将实际的反向梯度与"由编译后的前向
传播实际应用的掩码"所推算出的梯度进行比较（通过哪些输出元素恰好为零
来推断掩码）—— 二者不一致即证明反向传播重新抽取了与前向传播不同的
掩码，且无需借助 PyTorch 内部的调试工具。

## 本工具能做什么、不能做什么

这是一个**用户侧的调用边界缓解措施**，而不是对 PyTorch 分区器本身的
修复 —— 任何第三方软件包都无法从外部修补 AOTAutograd 内部的重新计算
节点选择逻辑。`safe_compile()` 会在被守护的编译调用期间临时将
`activation_memory_budget` 强制设为 1.0（唯一确认**不会**触发该缺陷
的取值），调用结束后（即使调用抛出异常）恢复调用方原先配置的预算值。
这是用"该次调用放弃 `budget<1.0` 带来的显存节省"换取"梯度正确性"的
明确权衡，而不是一个隐藏的权衡。

## 安装与诊断

需要 Python 3.9+ 及一个 PyTorch 构建。从源码安装：

```bash
git clone https://github.com/zhuhroscar-tech/torch-memory-budget-rng-guard.git
cd torch-memory-budget-rng-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-memory-budget-rng-guard
torch-memory-budget-rng-guard --json
```

如果你已经管理了一个兼容的 PyTorch 安装，可以不带 extra 直接安装 `.`。
使用 `--no-color` 输出纯文本。

退出码含义：**0** 表示 `safe_compile()` 在所有测试的预算值下都成功
阻止了梯度分歧（若未加防护的缺陷本身也没有复现，则打印"info"提示行，
而非警告），**1** 表示防护措施在至少一个测试的预算值下未能阻止该分歧，
**2** 表示无法导入 PyTorch。

## Python API

在计算图可能包含随机数算子、且调用方配置了
`activation_memory_budget < 1.0` 的任意 `torch.compile` 调用点上使用：

```python
from torch_memory_budget_rng_guard import safe_compile

# 替代:
#   compiled = torch.compile(model_fn)          # 若 budget<1.0 且随机
#   out = compiled(x)                            # 算子被重新计算，
#                                                  # 梯度可能被静默污染
guarded = safe_compile(model_fn)
out = guarded(x)   # 本次调用内强制 budget=1.0，结束后恢复你的设置
```

如果你需要 `budget<1.0` 带来的显存节省，并且已经独立验证过（例如针对
你自己安装的 torch 构建运行 `diagnose()`）你的具体计算图中不包含任何
随机数算子，可以绕过本防护措施，直接以你选择的预算值调用
`torch.compile`。

## 适用范围与限制

- 仅在 CPU 上复现和测试过（torch 2.14.0）。上游 issue 也报告了该缺陷
  在 CUDA 构建上存在，但本工具未独立在 CUDA/MPS 上验证。
- 该缓解措施较为粗粒度：它会在**整次**被守护的调用期间将预算强制设为
  1.0，而不是更精细的、针对随机数节点的预算策略。如果你的计算图完全
  不含随机数算子，本防护措施除了上下文管理器本身的开销外没有任何
  作用 —— 此时可以安全地跳过它。
- 不检测或防护在 `torch.compile` 区域**之外**使用的
  `randn_like`/其他随机数算子，也不防护未经过 `safe_compile()` 而由
  调用方直接编译的计算图。
- `diagnose()` 总是针对当前安装的 torch 构建重新运行实际复现流程 ——
  它从不假设某个特定的 PyTorch 版本受影响或不受影响。如果上游缺陷被
  修复并进入稳定版本，`any_unguarded_rng_recompute_bug` 会正确地报告
  为 `False`，此时本工具将成为一个无副作用的安全网，而非必需的
  workaround。

## 开发

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_memory_budget_rng_guard --cov-report=term-missing
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。
