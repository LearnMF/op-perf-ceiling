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

## 4. gfx936 / BW1000

待测：bw7 当前无法登录（主机名在 bw-zz 上解析不到，内网 IP 要密码）。拿到 gfx936 节点后用同样两条命令补上，
`tests/test_check.py` 会自动把新文件纳入回归。

## 5. 顺带发现并修掉的数据错误

`machines/gfx938-bw1102.json` 的 HBM「理论值」是 56 GB/s。它是 `calibrate` 用驱动报的 `memory_bus_width = 256`
算出来的，比实测 2015 GB/s 还低，导致 `--peaks theoretical` 下 137/138 个 shape 违例。
处理：删掉这几个理论值，并在 `calibrate` 里加了校验（理论值低于实测时不写入，只记一条 note）。
`tests/test_check.py::test_theoretical_peak_is_never_below_measured` 防止再出现。
