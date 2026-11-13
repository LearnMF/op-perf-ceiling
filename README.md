# op-perf-ceiling

**给定一张卡的硬件参数和一个算子的 shape，不上机就给出：这个算子在这张卡上最快能到多少、卡在哪种资源、离上限还差什么。**

面向 HCU/DCU（gfx936 / BW1000、gfx938 / BW1102，以及只有规格的未来卡）。方法依据是《HCU 算子性能优化教程》
第 3 章（Roofline）、第 4 章（分层 Roofline）、第 6 章（四个下限），本仓库把它做成工具。
代码从 [kernel-roofline-analysis-tools](https://github.com/LearnMF/kernel-roofline-analysis-tools)
（`feat/roofline-sim` @ 53fb4d0）迁出；那边保留 KDA 优化的历史工作和事后 profiling 工具。

## 输入 → 原理 → 流程 → 输出

**输入（三样）**

| 输入 | 是什么 | 在哪 |
|---|---|---|
| 机器文件 | CU 数、频率、各 dtype 算力峰值、HBM/L2/LDS 带宽、LDS/L2 容量、寄存器、延迟、launch 开销；每项存实测值和手册值 | `machines/*.json`（已有卡由 `opceil calibrate` 在真卡上生成；未知卡按同格式手写） |
| 算子 + shape + dtype + phase | 例如 `--op gemm --shape M=8192,N=4096,K=6144 --dtype bf16 --phase fwd` | 命令行 |
| 算子先验 | 必需计算量、必需字节、躲不开的其它工作、串行结构（L1 还有数据流和片上容量） | `opceil/ops/*.py` 插件，或 `opceil trace` 从 PyTorch 代码自动推导 |

**原理**：任何实现的耗时，都不可能少于「某项资源必须完成的工作量 ÷ 这项资源的峰值」。
每种资源（计算、HBM/L2/LDS、延迟、barrier、串行链、launch）各算一项；一个 kernel 内取最大，多个 kernel 串行相加；
片上放不下必须拆 kernel，拆出来的中间结果要多走 HBM。

**每一项都标了可信度**（这是读结果时最重要的一列）：

| 类别 | 含义 | 进入 |
|---|---|---|
| `bound` | 必需工作量 ÷ 峰值，可证明 | `T_bound`、`T_theory`、`T_split` |
| `model` | 依赖「kernel 怎么写/怎么调度」的假设：占用率、延迟掩藏、barrier、LDS 暂存、读写分开计价 | `T_theory`、`T_split` |
| `calibrated` | 用了按算子形状实测的机器常数（`structural.*`），换算子要重测 | `T_theory`、`T_split` |

**流程与输出**

| 层 | 含义 | 可信度 | 谁有 |
|---|---|---|---|
| `T_bound` | 只用 bound 项：**任何实现都打不破** | 可证明 | 所有算子 |
| `T_theory` | 完全融合、每字节只搬一次，加上 model / calibrated 项 | 取决于用到的项（报告里写明） | 所有算子 |
| `T_split` | 片上容量约束下，插件搜索空间内的最优拆法 | model 估计 | 有容量图的插件（目前 KDA） |
| `T_attain` | `T_split` × 实测空等系数 | 需标定 | 标定过的机器 + 算子族 |

报告（markdown，`--out` 时另出 JSON）给出：上述各层、瓶颈资源、算术强度与拐点、分层 roofline 表、MFU 上限、逐项时间及其类别；
有容量图的算子另有拆分搜索、逐轮融合台账、显存占用。保证 `T_bound <= T_theory <= T_split`（`tests/test_term_class.py`）。

## 用法

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e ".[dev]"   # + torch 可选
.venv/bin/python -m opceil predict --list-ops

# 1) 插件：按 shape 出上限
python -m opceil predict --machine machines/gfx938-bw1102.json --op gemm --shape M=8192,N=4096,K=6144 --dtype bf16
python -m opceil predict --machine machines/gfx936-bw1000.json --op attention --shape B=1,H=32,S=8192,D=128,causal=1

# 2) 有容量图的算子：拆分下限、融合台账、显存
python -m opceil predict --machine machines/gfx936-bw1000.json --op kda --shape T=16384,heads=96,tp=8 --phase fwd+bwd --split-search

# 3) 没有插件：从一段 PyTorch 代码自动推导（meta tensor，不需要卡）
python -m opceil trace --machine machines/gfx936-bw1000.json --script examples/torch/swiglu_mlp.py

# 4) 对真卡核对：卡上跑扫描，本机比对
HIP_VISIBLE_DEVICES=7 python bench/gemm_sweep.py --out gemm_bf16_gfx938-bw1102_<date>.jsonl
python -m opceil check --machine machines/gfx938-bw1102.json --measured data/measured/gemm_bf16_gfx938-bw1102_<date>.jsonl

# 5) 已有卡：生成/更新机器文件（在卡上）
python -m opceil calibrate --out machines/<arch>-<board>.json
```

## 现在有多准（2026-11-13）

GEMM 的 kernel 级核对（每张卡 138 个 bf16 shape，`docs/validation-gemm.md`）：

| | gfx936 / BW1000（bw52） | gfx938 / BW1102（nmz5） |
|---|---:|---:|
| 下限违例 | 0 / 138 | 0 / 138 |
| Spearman(T_bound, 实测) | 0.981 | 0.983 |
| η = T_bound / 实测，中位 | 0.67 | 0.57 |

- 大 shape 的算力受限、访存受限两端基本是紧的；
- **拐点附近和总工作量很小的 GEMM 偏乐观 2–8 倍**，两张卡规律一致（凹陷随拐点位置移动）：缺 tile 铺满 / 尾波项、拐点处的重叠项和 kernel 级最小时间项；
- 2 的幂方阵在两张卡上都明显变慢，疑似 HBM 通道冲突，待确认。

KDA：`T_bound` 0.605 ms（可证明），`T_theory` 0.837 ms（瓶颈是用 KDA 实测步长算的串行链，属 calibrated），
`T_split` 2.78 ms（model 估计，该拆法未被实现验证），生产实现 5.32 ms。

结论口径：**瓶颈类型和 `T_bound` 可以信；`T_theory` / `T_split` 的绝对值只在验证过的 shape 类别上可信。**

## 接入新算子

先读 [`docs/plugin-guide.md`](docs/plugin-guide.md)：判断走 L0 自动 / L0 插件 / L1 容量图哪一档，按必答清单逐项从算法推导，
标好每一项的类别，和 `opceil trace` 交叉核对。L0 插件样例：`opceil/ops/attention.py`；模板：`opceil/ops/_template.py`；
L1 样例：`opceil/ops/kda.py` + `kda_ir.py`。

## 目录

```
opceil/model/      求解器（与算子无关）：predict（下限与项类别）、capacity（拆分搜索）、ceiling、memory、rounds、check、validate
opceil/ops/        算子插件：gemm、elementwise / rmsnorm / softmax、attention（L0），kda（L1 容量图）
opceil/frontends/  torch meta 自动推导（opceil trace）
opceil/calibrate/  在真卡上生成机器文件的微基准
machines/          内置机器文件：gfx936-bw1000、gfx938-bw1102
bench/             卡上运行的测量脚本（只依赖 PyTorch）
data/              验证数据：kernel 级 GEMM 扫描、BW1000 训练 profile、KDA ceiling cards
reports/           opceil check 的核对报告
docs/              plugin-guide、validation-gemm；devlog/ 是迁移前的设计与里程碑记录
tests/             pytest
```
