# GEMM 上限对真卡的核对（kernel 级计时）

> 目的：在加任何新模型项之前，先确认「下限框不框得住真卡」和「离真卡多远」，并把差距按 shape 类别定位。
> 复现：
> ```
> # 卡上（容器内，只需要 PyTorch）
> HIP_VISIBLE_DEVICES=7 python bench/gemm_sweep.py --out gemm_bf16_<machine id>_<date>.jsonl
> # 本机
> python -m opceil check --machine machines/<machine id>.json --measured data/measured/gemm_bf16_<machine id>_<date>.jsonl
> ```
> 完整逐 shape 表：`reports/gemm_bf16_gfx938-bw1102_20261113.check.md`。

## 1. 为什么不用 hcu-train-simulator 的 BW1000 profile 做这件事

那份数据每条计时的是 Python 侧一整步（3 个 GEMM + cast/sum + autograd 调度），每条都带约 530 us 的地板，
下限几乎不可能被违反，「零违例」说明不了什么（见 `docs/devlog/m1-bw1000-validation.md`）。
`bench/gemm_sweep.py` 改成：单个 `torch.matmul`，`torch.profiler` 取设备上的 kernel 时间（BLAS 一次调用可能发两个
kernel，如 split-K + 归约，按调用求和），输入在多组缓冲区间轮换，避免 L2 常驻。

## 2. gfx938 / BW1102（nmz5，72 CU，torch 2.11 / HIP 7.2，2026-11-13，bf16，138 个 shape）

| | 值 |
|---|---:|
| 下限违例（T_bound > 1.01 × 实测） | **0 / 138** |
| η = T_bound / 实测：p10 / 中位 / p90 | 0.177 / 0.573 / 0.843 |
| Spearman(T_bound, 实测) | 0.983 |

η = 1 表示 kernel 跑在下限上，η = 0.5 表示用了下限的两倍时间。机器文件用实测峰值（mmac_bf16 371 TFLOPS，HBM 2015 GB/s）。

按 shape 类别（每个 (N,K) 系列扫 M = 1…8192）：

| 类别 | η | 读法 |
|---|---:|---|
| 大 shape 算力受限（min(M,N) > 2048） | p10 0.71 / 中位 0.83 / 最高 0.93 | **下限基本是紧的**：BLAS 中位跑到下限的 83%；个别例外见最后一行 |
| 大权重的 decode 型（M ≤ 16，N·K 大，访存受限） | 0.65–0.89 | **紧的**：权重读一遍，接近 HBM 实测峰值 |
| 拐点附近（M = 128–512） | 0.36–0.55 | **模型缺项**：`max(算力, 访存)` 假设两者同时满速且完全重叠，拐点处两边都不饱和 |
| 总工作量很小（N = 256 / 576，或 N·K 小；实测 9–25 us） | 0.12–0.25 | **模型缺项**：kernel 级最小时间约 9 us，远大于机器文件的 launch 2.2 us；tile 少，铺不满 72 个 CU，K 循环的串行延迟露出来 |
| 例外：方阵 8192³ / 16384³，N=6144,K=12288 的 M=2048/4096 | 0.72 / 0.59，0.52 / 0.52 | 反常（12288³ 是 0.93，同系列 M=8192 是 0.83）：疑似持续满载下降频或 BLAS 选到较差的 kernel 配置，待复测 |

## 3. 结论：现在能说什么、下一步补什么

能说的：
1. **下限成立**：138 个 shape 无一被打破，且排序几乎一致（Spearman 0.98）。
2. **大 shape 两端基本是紧的**：纯算力受限、纯访存受限的大 shape，真实 BLAS 多数达到下限的 65–93%（例外见上表）。

不能说的：
3. **拐点附近和小 shape 偏乐观 2–8 倍**。这两类正是「小 shape 乘 0.15 / 0.40」的分档系数想吸收的部分，原因是两个缺项：
   - **铺不满 + 尾波**：tile 数相对 CU 数太少或除不尽（需要 GEMM 的 tile 模型，tile 大小受 LDS / 寄存器约束）；
   - **kernel 级固定时间**：空 kernel 的 launch 2.2 us 不等于一个 K 循环 kernel 的最小时间（需要「最短 kernel」的微基准，按 K 步数计依赖延迟）。

   这两项都属于 model 类：会收紧 `T_theory`，但不改变 `T_bound`。

## 4. gfx936 / BW1000（bw52，80 CU，torch 2.11 / HIP 7.2，2026-11-13，bf16，同一组 138 个 shape）

| | 实测峰值档（476 TFLOPS / 1339 GB/s） | 手册峰值档（492 TFLOPS / 1843 GB/s） |
|---|---:|---:|
| 下限违例 | **0 / 138** | **0 / 138** |
| η：p10 / 中位 / p90 | 0.267 / 0.668 / 0.903 | 0.209 / 0.548 / 0.717 |
| Spearman | 0.981 | 0.984 |

完整逐 shape 表：`reports/gemm_bf16_gfx936-bw1000_20261113.check.md`。

### 两张卡对照：规律一致，所以是系统性的

| 现象 | gfx936 | gfx938 | 读法 |
|---|---|---|---|
| 大 shape 两端 | min(M,N)>2048 中位 0.76、最高 0.87；大权重 decode 型（M≤16）多数 0.66–0.95 | min(M,N)>2048 中位 0.83、最高 0.93；decode 型 0.65–0.89 | 下限基本是紧的 |
| 拐点凹陷 | 出现在 **M≈512**（η 0.45–0.47） | 出现在 **M≈256**（η 0.36–0.44） | 凹陷位置随拐点移动（gfx936 拐点 355、gfx938 拐点 184 FLOP/B）：`max(算力, 访存)` 的完全重叠假设在拐点处偏乐观，是模型缺项，不是某张卡的偶然 |
| 小 shape | η 0.2–0.4，最短 kernel 7.3 us | η 0.12–0.25，最短 kernel 9.0 us | kernel 级最小时间 + 铺不满 CU，模型缺项 |
| 2 的幂方阵 | 8192³ 0.65、16384³ 0.55；同配置的 12288³ 0.85 | 8192³ 0.72、16384³ 0.59；12288³ 0.93 | 两张卡同样掉、BLAS 选的是同一种 256×256 tile：**疑似 2 的幂步长造成 HBM 通道冲突**（假设，待用非 2 的幂 leading dimension 复测确认） |
| M=1 单点 | N=6144 系列 M=1 比 M=2 慢近一倍（0.47 vs 0.84，0.60 vs 0.94） | 无此现象 | BLAS 在 gfx936 上对 M=1 选了较差的路径：是库的问题，不是模型的 |

## 5. 顺带发现并修掉的数据错误

`machines/gfx938-bw1102.json` 的 HBM「理论值」是 56 GB/s。它是 `calibrate` 用驱动报的 `memory_bus_width = 256`
算出来的，比实测 2015 GB/s 还低，导致 `--peaks theoretical` 下 137/138 个 shape 违例。
处理：删掉这几个理论值，并在 `calibrate` 里加了校验（理论值低于实测时不写入，只记一条 note）。
`tests/test_check.py::test_theoretical_peak_is_never_below_measured` 防止再出现。

## 6. 补 GEMM 分块模型之后（2026-11-13，第二轮）

### 6.1 加了什么

`opceil/model/tiling.py`：把一个 GEMM 当成「输出 tile 网格」来计价，在候选空间（tile 大小、split-K、线程数、LDS 级数）里
挑出放得进 LDS / 寄存器的最快一种，作为 `T_split`（model 类，不进 `T_bound`）。新增的项：

| 项 | 含义 |
|---|---|
| `compute:tiles` | 补零后的计算量 ÷（网格能占满的那部分 CU 的算力）：尾波和小网格的损失 |
| `memory:hbm`（按 CTA 数） | 网格只有这么多 CTA 时能拉到的带宽（机器文件的 CTA–带宽曲线，形状来自微基准，饱和值取流式峰值） |
| `memory:l2` / `memory:lds` | A 按 tile 列数、B 按 tile 行数重读 |
| `serial:kloop` | 每波 CTA：首次 HBM 延迟 + 每个 K 步 max(依赖 MMAC + barrier, 操作数延迟 ÷ 预取深度) + 写回 L2 延迟 |
| `fixed:launch` | 取 dispatch 间隔与「平凡 kernel 最短设备时长」（`bench/kernel_floor.py`，两张卡约 1.6 / 1.9 us）较大者 |

所有假设登记在 `docs/devlog/constant-ledger.md`；**没有用这 276 个实测点调过任何常数**。

### 6.2 效果：只收回了一小部分差距

`T_theory`（融合、实测峰值）和 `T_split`（分块模型）各自 ÷ 实测，中位数：

| min(M,N) | gfx936 T_theory → T_split | gfx938 T_theory → T_split |
|---|---:|---:|
| ≤ 16 | 0.82 → 0.82 | 0.67 → 0.67 |
| 17–256 | 0.62 → 0.63 | 0.44 → 0.49 |
| 257–2048（拐点附近） | 0.48 → **0.57** | 0.55 → **0.61** |
| > 2048 | 0.76 → 0.79 | 0.83 → 0.83 |
| 全部 | 0.66 → 0.68 | 0.57 → 0.60 |

两张卡上 `T_split` 都**零违例**：模型没有比真实库算得更贵。但收紧有限，主要在拐点附近。
小 shape 几乎没动：模型认为存在一个 2–4 us 的分块方案（小 tile + 大 split-K + 4 级预取），实测库要 9–25 us。

### 6.3 差距是不是「库没选好」：TunableOp 对照

`bench/gemm_sweep.py --tunableop` 让 PyTorch TunableOp 把每个 shape 的所有 hipBLASLt / rocBLAS 解都试一遍取最快
（≤ 50 GFLOP 的 104 个 shape，数据在 `data/measured/*_tunableop.jsonl`）：

| | gfx936 | gfx938 |
|---|---:|---:|
| 比默认选择快 10% 以上的 shape | 15 / 104 | 23 / 104 |
| 最大提速 | 1.69x（M=1, N=6144, K=2048） | 1.58x（同一 shape） |
| `T_split` / 实测，中位：默认 → 库内最优 | 0.64 → 0.69 | 0.55 → 0.56 |

结论：库的启发式选择**只解释了一小部分**（集中在 decode 型小 M）。剩下的差距，库里没有任何现成解能填上：
要么是手写 kernel 能拿到的空间，要么是模型漏了硬件行为。**用 BLAS 当尺子分不开这两者**；要分开，只能针对某一类 shape
写一个 kernel 去逼近 `T_split`。这是下一步该做的实验，不是继续往模型里加项。

### 6.4 2 的幂方阵：确认一半

`bench/ld_pad.py`：同一个 GEMM，把 leading dimension 从 2 的幂加 64 个元素（`data/measured/ld_pad_20261113.json`）：

| | 8192³ | 16384³ | 对照 4096³ / 12288³ |
|---|---:|---:|---|
| gfx936 | 308 → **399** TFLOPS（+30%） | 263 → **399**（+52%） | 不变 / 略降 |
| gfx938 | 266 → **311**（+17%） | 220 → 220（不变） | 不变 / 略降 |

gfx936 上就是 2 的幂步长的访存冲突，**补齐 leading dimension 即可拿回**（可直接作为优化建议）；
gfx938 的 16384³ 另有原因（疑似持续满载降频），未确认。这类地址映射效应目前不在模型里。

### 6.5 顺带修正的两处机器数据

1. **gfx936 的 HBM 实测峰值偏低**：TunableOp 选出的 hipBLASLt kernel 在 M=32, N=24576, K=6144 上读 B 达到 1365.6 GB/s，
   高于带宽微基准的 1339.2 GB/s，导致当时的 `T_bound` 被打破 2%。处理：
   - **`T_bound` 一律改用手册峰值**（机器文件有 `theoretical` 时），因为实测峰值只是「微基准跑到的最好值」，是真实峰值的下估；
     没有手册值的卡（gfx938）报告里会写明「T_bound 对哪些峰值退化为实测值」；
   - `peaks.hbm_read.value` 提到 1365.6（原值留在 `microbench_value`，来源写进 `_observed`）：实测峰值 = 任何 kernel 达到过的最好值；
   - CTA–带宽曲线只取形状，饱和值按流式峰值缩放（gfx938 的曲线顶点 1909 原本低于自身的流式峰值 2015）。
2. 第 5 节的 gfx938 理论带宽 56 GB/s 问题（已修）。
