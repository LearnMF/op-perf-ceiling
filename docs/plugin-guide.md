# 怎么给一个新算子接入上限预测（插件编写指南）

> 适用对象：要给新算子出上限的人。读完应能判断「要不要写插件、写到哪一档、每个数从哪来」。
> 方法论依据是《HCU 算子性能优化教程》第 3 章（Roofline）、第 4 章（分层 Roofline）、第 6 章（四个下限）；
> 本工具就是把这三章做成程序。KDA 的**方法**可以照搬，KDA 的**数字**（递推步长、tile 效率、寄存器基数）不能搬。

## 1. 先选档位：不一定要写插件

| 你要回答的问题 | 档位 | 怎么做 | 工作量 |
|---|---|---|---|
| 一段 PyTorch 代码的理论上限、瓶颈类型、eager 和融合差多少 | **L0 自动** | `python -m opceil trace --script my_block.py`，不写插件 | 写一个 `build()` |
| 同一类算子要按 shape 反复出数、给别人用 | **L0 插件** | 照 `opceil/ops/attention.py` 写一个类 | 半天 |
| 片上放不下要拆几个 kernel、哪些该融合、拆分代价多大 | **L1 容量图插件** | 照 `opceil/ops/kda.py` 的 `capacity_graph()` 写数据流图 | 数天，且要实测常数 |

L0 能给：`T_bound`（可证明下限）、算术强度、拐点、瓶颈在算力侧还是访存侧、MFU 天花板。
L0 不能给：拆分下限 `T_split`、融合建议、片上容量造成的上限损失——这些只有 L1 有。

### L0 自动（torch meta）的边界
- 在 `device="meta"` 上跑，只有形状没有数据，笔记本上就能跑整个 block；前向 + `.backward()` 都会被记录。
- 计算量只对有 torch flop 公式的算子计数（mm/bmm/addmm/conv/SDPA）；elementwise、归约记 0 FLOP，只计字节。
- **直接从 Python 发射的 Triton kernel（如 FLA）不经过 dispatcher，也不能在 meta tensor 上跑**：要换成它的纯 PyTorch 参考实现再 trace。
- 输出里 `fused` = 整段当一个 kernel（只有函数自己的输入输出过 HBM），`eager` = 每个 aten op 一个 kernel；两者之比是融合最多能拿回的倍数。

## 2. 必答清单（每个插件都要回答，L1 多一项）

每一项都必须**从算法推出来**，不能从某个已有 kernel 的实测里抄。

| # | 问题 | 填到 `Work` 的哪里 | 项的类别 |
|---|---|---|---|
| 1 | 必需计算量：矩阵乘加次数 × 2 | `flops` | bound |
| 2 | 必需 HBM 字节：输入各读一次 + 输出各写一次（训练前向还要加反向要用的保存张量） | `bytes_min`（或 `bytes_rd` + `bytes_wr`，后者按读写分别计价，属 model） | bound |
| 3 | 其它躲不开的工作：超越函数次数、只能走向量单元的 lane-op | `other["sfu_exp2"]` 等；数量依赖实现假设的（如打包比例）同时写进 `other_model` | bound / model |
| 4 | 串行结构：不能并行掉的依赖步（scan、递推） | `serial=[(步数, 机器文件里的步长键, CTA 数)]` | 用通用依赖延迟 = bound；用算子专属实测步长 = calibrated |
| 5 | （L1）数据流：有哪些 op、op 之间传哪些张量、每个 tile 在片上要放多少 | `capacity_graph()` 返回 `CapGraph` | model |

「项的类别」决定它进不进 `T_bound`：
- **bound**：必需工作量 ÷ 峰值，可证明，进 `T_bound`；
- **model**：依赖「kernel 会怎么写/怎么调度」的假设（占用率、延迟掩藏、barrier、LDS 暂存、读写分开计价），只进 `T_theory`/`T_split`；
- **calibrated**：用了按算子形状测出的机器常数（`structural.*`），只进 `T_theory`/`T_split`，换算子必须重测。

报告里每一列都标了类别；`T_bound <= T_theory <= T_split` 由测试保证。

## 3. 写一个 L0 插件：以 attention 为例

1. 复制 `opceil/ops/_template.py` 到 `opceil/ops/<name>.py`，改 `name` / `params` / `phases`。
2. 在 `work()` 里按第 2 节清单逐项填。`attention.py` 的答案是：
   - FLOPs：`QK^T` 和 `PV` 各一次乘加 → `4 × 计算的 (q,k) 对数 × D`；causal 时对数按下三角精确计；
   - 字节：Q、K、V 各读一次（GQA 时 K/V 按 `Hkv` 计），O 写一次，加 fp32 的 lse；
   - 超越函数：每个分数一次 exp → `other["sfu_exp2"]`；
   - 串行：无。
3. 在 `opceil/ops/__init__.py` 的 `registry()` 里把模块加进列表。
4. 写测试，至少两条：
   - **和 torch 自动推导对一遍**（`tests/test_torch_meta.py::test_attention_flops_match_the_plugin`）：两条路给出的 FLOPs/字节应一致，不一致说明有一边漏了；
   - 典型 shape 的瓶颈侧（访存/算力）符合预期。
5. 跑：`python -m opceil predict --machine machines/gfx936-bw1000.json --op attention --shape B=1,H=32,S=8192,D=128,causal=1`

## 4. 写一个 L1 容量图插件：KDA 是唯一样例

只有当问题是「拆成几个 kernel、怎么融合」时才写。需要给出（字段定义见 `opceil/model/capacity.py`）：

| 结构 | 字段 | 含义 | KDA 里怎么来的 |
|---|---|---|---|
| `TileTensor` | `tile_bytes` / `hbm_bytes` / `kind` / `residency` | 一个张量在**一个 tile** 里占多少片上字节、整体过 HBM 多少字节、是输入/输出/中间量、放寄存器还是 LDS | `kda_ir.py` 的逻辑张量表 + chunk 大小 |
| `CapOp` | `ins` / `outs` / `barrier` / `core_lds` / `core_reg_bytes` / `movable` / `ctas` / `flops` / `serial` | 一个子 op 读写哪些张量、并行结构是否和邻居不同（不同就不能同 kernel）、自身乘积要多少 LDS/寄存器、网格多大 | `kda_ir.py` 的 op 表，每个 op 标了 `par`（row/chunk/scan） |
| `CapGraph` | `ops` / `tensors` / `threads` / `tiles` / `reg_base_vgpr` | 整个数据流 + 每 tile 线程数 + tile 总数 | 一个 (chunk, head) 一个 tile，256 线程 |
| 插件方法 | `capacity_graphs(shape, phase, residency, vgpr)` | 产出每种算法形式（重算 vs 存状态等）各一张图 | `kda.py` 的 `FORMULATIONS` |

求解器（与算子无关）会：枚举所有连续分组 → 用机器文件的 LDS 容量和寄存器上限判可行 → 放不下的中间量落 HBM → 每种拆法用同一套资源项计价 → 给出 `T_split`、逐轮融合台账、显存。

写 L1 时要警惕的三件事（都来自 KDA 的教训，见 `docs/devlog/constant-ledger.md`）：
1. **不要拿优化后的 kernel 的实测占用去填 `core_lds`/寄存器**——那是拟合答案，只能拿来事后对照；
2. 拍定的启发式常数（如 KDA 的 `REG_BASE_VGPR = 16`、`VALU_PACKED = 0.75`）要在 `constant-ledger.md` 登记，并在 `other_model` / 文档里标为 model；
3. 需要实测的算子专属常数（递推步长、tile 效率）写进机器文件 `structural`，报告会把用到它的项标为 calibrated。

## 5. 机器文件：已有卡和未知卡

插件与机器无关，同一个插件在任何机器文件上都能跑：
- **已有卡**（`machines/gfx936-bw1000.json`、`machines/gfx938-bw1102.json`）：`python -m opceil calibrate` 在真卡上跑微基准生成，每项同时存实测值和手册值（`--peaks measured|theoretical` 切换）；
- **未知卡**（S4000/S2000 等）：按同一格式手写，数据来自规格书或 SOC 仿真；缺的字段不猜，报告头给出能力等级（L0–L3）和缺哪些字段，见 `docs/devlog/machine-capability-levels.md`。

## 6. 提交前自查

- [ ] 第 2 节每一项都写了来源（算法推导 / 机器文件哪个键），没有从实测 kernel 抄数；
- [ ] 依赖实现假设的量放进了 `other_model` 或登记为 model；
- [ ] 有和 `opceil trace` 的交叉核对（标准 PyTorch 能表达的算子）；
- [ ] `T_bound <= 实测` 在手头能测的 shape 上成立（`tests/test_term_class.py` 的参数表里加上这个算子）。
