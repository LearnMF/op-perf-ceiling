# opceil check: GEMM T_bound vs kernel-level measurement

machine file `gfx938-bw1102` (measured peaks: mmac_bf16 371 TFLOPS, HBM 2020 GB/s); measured on nmz5 BW1102 (72 CUs), torch 2.11.0, 2026-11-13 17:56:04

| | value |
| --- | ---: |
| shapes | 104 |
| **bound violations** (T_bound > 1.01 x measured) | **0** |
| eta = T_bound / measured: p10 / median / p90 | 0.153 / 0.524 / 0.834 |
| Spearman(T_bound, measured) | 0.923 |
| T_split (tiled, model) above measured (> 1.01 x) | 0 |
| eta_split = T_split / measured: p10 / median / p90 | 0.165 / 0.555 / 0.872 |
| Spearman(T_split, measured) | 0.925 |

eta = 1 means the measured kernel runs at the bound; eta = 0.5 means it takes twice the bound.  T_split above measured means the tiled model claims more cost than the vendor BLAS actually pays: the model is wrong there, not the library.

## by bottleneck side of the bound

| side | n | p10 | median | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| compute | 30 | 0.203 | 0.439 | 0.708 | 0.792 |
| fixed | 6 | 0.140 | 0.143 | 0.206 | 0.206 |
| memory | 68 | 0.162 | 0.609 | 0.934 | 0.959 |

## by size (T_bound | T_split)

| min(M,N) | n | eta p10 | eta median | eta p90 | eta_split p10 | eta_split median | eta_split p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| min(M,N) <= 16 | 40 | 0.163 | 0.677 | 0.942 | 0.163 | 0.679 | 0.944 |
| 17-256 | 44 | 0.137 | 0.413 | 0.701 | 0.153 | 0.453 | 0.703 |
| 257-2048 | 18 | 0.206 | 0.516 | 0.685 | 0.280 | 0.567 | 0.770 |
| > 2048 | 2 | 0.708 | 0.719 | 0.730 | 0.708 | 0.719 | 0.730 |

## all shapes (sorted by eta)

| M | N | K | measured us | T_bound us | eta | side | T_split us | eta_split | split binding | tile | TFLOPS | GB/s | kernels |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 2 | 4096 | 4096 | 326.17 | 16.63 | 0.051 | memory | 16.67 | 0.051 | memory:hbm | 16x16x64/s1 | 0.2 | 103 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 64 | 4096 | 11008 | 736.13 | 45.61 | 0.062 | memory | 46.33 | 0.063 | memory:hbm | 32x128x16/s2 | 7.8 | 125 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 64 | 256 | 6144 | 15.22 | 1.96 | 0.129 | memory | 2.01 | 0.132 | memory:hbm | 32x128x16/s32 | 13.2 | 261 | Cijk_Ailk_Bljk_BBH_MT32x64x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 32 | 256 | 6144 | 14.19 | 1.92 | 0.135 | fixed | 1.92 | 0.135 | fixed:launch | 16x32x64/s8 | 7.1 | 251 | Cijk_Ailk_Bljk_BBH_MT64x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 256 | 4096 | 11008 | 457.82 | 62.18 | 0.136 | compute | 69.96 | 0.153 | compute:mmac_bf16 | 128x256x16/s2 | 50.4 | 214 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1, Cijk_B_PostGSU x1 |
| 128 | 256 | 6144 | 17.29 | 2.37 | 0.137 | memory | 2.54 | 0.147 | memory:hbm | 64x128x32/s16 | 23.3 | 277 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 8 | 256 | 6144 | 13.70 | 1.92 | 0.140 | fixed | 1.92 | 0.140 | fixed:launch | 16x16x32/s32 | 1.8 | 237 | Cijk_Ailk_Bljk_BBH_MT16x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 1 | 256 | 6144 | 13.61 | 1.92 | 0.141 | fixed | 1.92 | 0.141 | fixed:launch | 16x16x32/s32 | 0.2 | 232 | Cijk_Ailk_Bljk_BBH_MT16x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 2 | 256 | 6144 | 13.23 | 1.92 | 0.145 | fixed | 1.92 | 0.145 | fixed:launch | 16x16x32/s32 | 0.5 | 240 | Cijk_Ailk_Bljk_BBH_MT16x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 256 | 256 | 6144 | 20.98 | 3.18 | 0.152 | memory | 3.46 | 0.165 | memory:hbm | 128x128x16/s16 | 38.4 | 306 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 128 | 576 | 4608 | 21.41 | 3.29 | 0.153 | memory | 3.60 | 0.168 | memory:l2 | 128x64x32/s8 | 31.7 | 310 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 512 | 256 | 6144 | 30.26 | 4.80 | 0.159 | memory | 5.38 | 0.178 | memory:l2 | 128x256x16/s16 | 53.2 | 321 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 2048 | 576 | 4608 | 183.41 | 29.28 | 0.160 | compute | 36.60 | 0.200 | compute:tiles | 256x128x16/s8 | 59.3 | 145 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1 |
| 256 | 576 | 4608 | 24.35 | 3.94 | 0.162 | memory | 6.17 | 0.253 | memory:l2 | 256x64x16/s8 | 55.8 | 327 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 16 | 256 | 6144 | 11.75 | 1.92 | 0.163 | fixed | 1.92 | 0.163 | fixed:launch | 16x16x32/s32 | 4.3 | 285 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 64 | 576 | 4608 | 17.97 | 2.96 | 0.165 | memory | 3.05 | 0.170 | memory:hbm | 64x64x16/s32 | 18.9 | 332 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 32 | 576 | 4608 | 15.44 | 2.79 | 0.181 | memory | 2.84 | 0.184 | memory:hbm | 16x64x16/s16 | 11.0 | 365 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 512 | 576 | 4608 | 39.27 | 7.32 | 0.186 | compute | 10.98 | 0.280 | compute:tiles | 128x128x16/s8 | 69.2 | 270 | Cijk_Ailk_Bljk_BBH_MT64x64x64_SE_AMAS3_B x1 |
| 8 | 576 | 4608 | 13.61 | 2.67 | 0.196 | memory | 2.69 | 0.197 | memory:hbm | 16x16x16/s32 | 3.1 | 396 | Cijk_Ailk_Bljk_BBH_MT64x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 256 | 2304 | 1152 | 18.06 | 3.66 | 0.203 | compute | 5.19 | 0.287 | memory:l2 | 128x256x16/s4 | 75.2 | 392 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 512 | 512 | 512 | 9.32 | 1.92 | 0.206 | fixed | 1.92 | 0.206 | fixed:launch | 64x64x64/s1 | 28.8 | 169 | Cijk_Ailk_Bljk_BBH_MT128x32x64_SE_AMAS3_ x1 |
| 16 | 576 | 4608 | 13.14 | 2.71 | 0.206 | memory | 2.74 | 0.208 | memory:hbm | 16x16x16/s32 | 6.5 | 416 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 2 | 576 | 4608 | 12.71 | 2.64 | 0.208 | memory | 2.65 | 0.208 | memory:hbm | 16x16x16/s32 | 0.8 | 419 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 1 | 576 | 4608 | 12.57 | 2.63 | 0.209 | memory | 2.64 | 0.210 | memory:hbm | 16x16x16/s32 | 0.4 | 423 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 128 | 2304 | 1152 | 14.50 | 3.07 | 0.212 | memory | 3.76 | 0.259 | memory:hbm | 64x128x32/s2 | 46.9 | 427 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 1024 | 256 | 6144 | 39.76 | 8.68 | 0.218 | compute | 9.76 | 0.246 | compute:mmac_bf16 | 128x256x16/s8 | 81.0 | 409 | Cijk_Ailk_Bljk_BBH_MT64x64x64_SE_AMAS3_B x1 |
| 8 | 2304 | 1152 | 11.90 | 2.66 | 0.223 | memory | 2.66 | 0.224 | memory:hbm | 16x16x128/s1 | 3.6 | 451 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 128 | 4096 | 11008 | 206.96 | 46.57 | 0.225 | memory | 47.90 | 0.231 | memory:hbm | 64x256x16/s4 | 55.8 | 454 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 1 | 2304 | 1152 | 11.61 | 2.63 | 0.227 | memory | 2.64 | 0.227 | memory:hbm | 16x16x128/s1 | 0.5 | 458 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 16 | 2304 | 1152 | 11.64 | 2.68 | 0.231 | memory | 2.69 | 0.231 | memory:hbm | 16x16x128/s1 | 7.3 | 466 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 2 | 2304 | 1152 | 11.20 | 2.64 | 0.235 | memory | 2.64 | 0.236 | memory:hbm | 16x16x128/s1 | 0.9 | 475 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 64 | 2304 | 1152 | 11.39 | 2.85 | 0.250 | memory | 3.17 | 0.278 | memory:l2 | 32x64x64/s1 | 29.8 | 505 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 32 | 2304 | 1152 | 10.88 | 2.74 | 0.252 | memory | 2.75 | 0.253 | memory:hbm | 16x64x64/s1 | 15.6 | 508 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 1024 | 576 | 4608 | 55.21 | 14.64 | 0.265 | compute | 20.98 | 0.380 | memory:l2 | 256x128x16/s16 | 98.5 | 288 | Cijk_Ailk_Bljk_BBH_MT128x96x32_SE_AMAS3_ x1 |
| 1024 | 1024 | 1024 | 20.42 | 5.78 | 0.283 | compute | 6.97 | 0.341 | memory:l2 | 128x128x16/s1 | 105.2 | 308 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 512 | 2304 | 1152 | 25.25 | 7.32 | 0.290 | compute | 8.30 | 0.329 | memory:l2 | 256x256x16/s4 | 107.6 | 350 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 32 | 4096 | 11008 | 144.66 | 45.13 | 0.312 | memory | 45.25 | 0.313 | memory:hbm | 16x64x32/s1 | 19.9 | 630 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 2048 | 256 | 6144 | 54.77 | 17.35 | 0.317 | compute | 19.52 | 0.356 | compute:mmac_bf16 | 128x256x16/s4 | 117.6 | 536 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 256 | 6144 | 2048 | 48.62 | 17.35 | 0.357 | compute | 20.29 | 0.417 | memory:l2 | 128x256x16/s4 | 132.5 | 604 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 1024 | 2304 | 1152 | 40.60 | 14.64 | 0.361 | compute | 14.64 | 0.361 | compute:mmac_bf16 | 128x256x16/s1 | 133.9 | 305 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 4096 | 4096 | 58.26 | 23.14 | 0.397 | compute | 26.03 | 0.447 | compute:mmac_bf16 | 128x256x16/s2 | 147.5 | 648 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 4096 | 576 | 4608 | 147.44 | 58.57 | 0.397 | compute | 69.18 | 0.469 | memory:l2 | 256x128x16/s8 | 147.5 | 324 | Cijk_Ailk_Bljk_BBH_MT128x192x16_SE_AMAS3 x1 |
| 256 | 6144 | 12288 | 253.20 | 104.12 | 0.411 | compute | 107.37 | 0.424 | compute:tiles | 256x256x16/s32 | 152.7 | 634 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 4096 | 6144 | 83.53 | 34.71 | 0.416 | compute | 39.05 | 0.467 | compute:mmac_bf16 | 128x256x16/s2 | 154.3 | 665 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 128 | 6144 | 2048 | 30.90 | 13.50 | 0.437 | memory | 15.06 | 0.488 | memory:l2 | 64x128x16/s1 | 104.3 | 882 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 128 | 4096 | 6144 | 59.95 | 26.22 | 0.437 | memory | 27.51 | 0.459 | memory:hbm | 64x256x16/s4 | 107.5 | 883 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 128 | 4096 | 4096 | 39.75 | 17.65 | 0.444 | memory | 18.92 | 0.476 | memory:hbm | 64x256x16/s4 | 108.0 | 897 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 4096 | 256 | 6144 | 75.09 | 34.71 | 0.462 | compute | 39.05 | 0.520 | compute:mmac_bf16 | 128x256x16/s2 | 171.6 | 740 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 8192 | 256 | 6144 | 141.06 | 69.41 | 0.492 | compute | 78.09 | 0.554 | compute:mmac_bf16 | 128x256x16/s1 | 182.7 | 766 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 512 | 4096 | 6144 | 136.03 | 69.41 | 0.510 | compute | 78.09 | 0.574 | compute:mmac_bf16 | 128x256x16/s1 | 189.4 | 447 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 128 | 11008 | 4096 | 89.93 | 46.57 | 0.518 | memory | 49.97 | 0.556 | memory:hbm | 64x256x16/s2 | 128.3 | 1046 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 512 | 4096 | 4096 | 88.74 | 46.28 | 0.521 | compute | 52.06 | 0.587 | compute:mmac_bf16 | 128x256x16/s1 | 193.6 | 473 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 64 | 4096 | 6144 | 48.64 | 25.57 | 0.526 | memory | 26.24 | 0.540 | memory:hbm | 32x128x16/s2 | 66.2 | 1062 | Cijk_Ailk_Bljk_BBH_MT256x64x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 512 | 6144 | 2048 | 65.42 | 34.71 | 0.531 | compute | 39.05 | 0.597 | compute:tiles | 128x256x16/s2 | 197.0 | 513 | Cijk_Ailk_Bljk_BBH_MT128x192x16_SE_AMAS3 x1 |
| 512 | 4096 | 11008 | 234.31 | 124.37 | 0.531 | compute | 132.82 | 0.567 | memory:l2 | 128x256x16/s16 | 197.1 | 451 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 64 | 4096 | 4096 | 32.13 | 17.13 | 0.533 | memory | 17.79 | 0.554 | memory:hbm | 32x128x16/s2 | 66.8 | 1077 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 8192 | 576 | 4608 | 219.48 | 117.14 | 0.534 | compute | 131.78 | 0.600 | compute:tiles | 256x128x16/s4 | 198.1 | 411 | Cijk_Ailk_Bljk_BBH_MT256x384x16_SE_AMAS3 x1 |
| 256 | 11008 | 4096 | 116.36 | 62.18 | 0.534 | compute | 65.08 | 0.559 | compute:tiles | 128x256x16/s4 | 198.4 | 841 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 128 | 6144 | 12288 | 142.71 | 77.10 | 0.540 | memory | 79.12 | 0.554 | memory:hbm | 64x256x16/s2 | 135.4 | 1091 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 512 | 11008 | 4096 | 229.24 | 124.37 | 0.543 | compute | 130.15 | 0.568 | compute:tiles | 128x256x16/s2 | 201.4 | 461 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 64 | 6144 | 2048 | 23.29 | 12.98 | 0.557 | memory | 13.06 | 0.561 | memory:hbm | 32x128x16/s1 | 69.1 | 1125 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 128 | 24576 | 6144 | 275.08 | 153.43 | 0.558 | memory | 154.16 | 0.560 | memory:hbm | 64x256x16/s1 | 140.5 | 1126 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 8 | 4096 | 11008 | 79.48 | 44.77 | 0.563 | memory | 44.88 | 0.565 | memory:hbm | 16x16x64/s1 | 9.1 | 1138 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 64 | 11008 | 4096 | 77.57 | 45.61 | 0.588 | memory | 45.80 | 0.590 | memory:hbm | 32x128x16/s1 | 74.4 | 1187 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1 |
| 32 | 4096 | 4096 | 26.77 | 16.87 | 0.630 | memory | 16.93 | 0.632 | memory:hbm | 16x64x32/s1 | 40.1 | 1273 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 1 | 6144 | 2048 | 19.52 | 12.47 | 0.639 | memory | 12.50 | 0.640 | memory:hbm | 16x16x32/s1 | 1.3 | 1290 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 32 | 6144 | 2048 | 19.37 | 12.72 | 0.657 | memory | 12.77 | 0.660 | memory:hbm | 16x64x16/s1 | 41.6 | 1326 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 8 | 6144 | 2048 | 19.06 | 12.53 | 0.657 | memory | 12.56 | 0.659 | memory:hbm | 16x16x32/s1 | 10.6 | 1327 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 2 | 6144 | 2048 | 18.94 | 12.48 | 0.659 | memory | 12.51 | 0.660 | memory:hbm | 16x16x32/s1 | 2.7 | 1330 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8 | 4096 | 4096 | 25.23 | 16.68 | 0.661 | memory | 16.72 | 0.663 | memory:hbm | 16x16x64/s1 | 10.6 | 1335 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 16 | 4096 | 4096 | 25.18 | 16.75 | 0.665 | memory | 16.79 | 0.667 | memory:hbm | 16x16x64/s1 | 21.3 | 1343 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 1 | 4096 | 4096 | 24.99 | 16.62 | 0.665 | memory | 16.66 | 0.667 | memory:hbm | 16x16x64/s1 | 1.3 | 1343 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 1024 | 6144 | 2048 | 101.37 | 69.41 | 0.685 | compute | 78.09 | 0.770 | compute:tiles | 128x256x16/s1 | 254.2 | 414 | Cijk_Ailk_Bljk_BBH_MT256x384x16_SE_AMAS3 x1 |
| 2048 | 2304 | 1152 | 42.75 | 29.28 | 0.685 | compute | 29.28 | 0.685 | compute:mmac_bf16 | 128x256x16/s1 | 254.3 | 455 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 16 | 6144 | 2048 | 18.28 | 12.59 | 0.689 | memory | 12.63 | 0.691 | memory:hbm | 16x16x32/s1 | 22.0 | 1391 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 32 | 4096 | 6144 | 36.58 | 25.25 | 0.690 | memory | 25.32 | 0.692 | memory:hbm | 16x64x32/s1 | 44.0 | 1394 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 64 | 6144 | 12288 | 108.26 | 75.94 | 0.701 | memory | 76.16 | 0.703 | memory:hbm | 32x128x16/s1 | 89.3 | 1416 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 4096 | 2304 | 1152 | 82.72 | 58.57 | 0.708 | compute | 58.57 | 0.708 | compute:mmac_bf16 | 128x256x16/s1 | 262.8 | 406 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 2048 | 2048 | 2048 | 65.18 | 46.28 | 0.710 | compute | 52.06 | 0.799 | compute:tiles | 128x256x16/s1 | 263.6 | 386 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 32 | 11008 | 4096 | 63.11 | 45.13 | 0.715 | memory | 45.28 | 0.717 | memory:hbm | 16x64x16/s1 | 45.7 | 1444 | Cijk_Ailk_Bljk_BBH_MT256x32x32_SE_AMAS3_ x1 |
| 16 | 4096 | 6144 | 34.88 | 25.09 | 0.719 | memory | 25.15 | 0.721 | memory:hbm | 16x16x64/s1 | 23.1 | 1452 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 8192 | 2304 | 1152 | 160.56 | 117.14 | 0.730 | compute | 117.14 | 0.730 | compute:mmac_bf16 | 128x256x16/s1 | 270.8 | 386 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 64 | 24576 | 6144 | 207.32 | 151.48 | 0.731 | memory | 152.03 | 0.733 | memory:hbm | 32x128x16/s1 | 93.2 | 1476 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 8 | 4096 | 6144 | 33.07 | 25.00 | 0.756 | memory | 25.07 | 0.758 | memory:hbm | 16x16x64/s1 | 12.2 | 1527 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 1 | 4096 | 6144 | 32.71 | 24.93 | 0.762 | memory | 24.99 | 0.764 | memory:hbm | 16x16x64/s1 | 1.5 | 1540 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 2 | 4096 | 6144 | 32.66 | 24.94 | 0.764 | memory | 25.00 | 0.766 | memory:hbm | 16x16x64/s1 | 3.1 | 1543 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 16 | 4096 | 11008 | 58.07 | 44.89 | 0.773 | memory | 45.01 | 0.775 | memory:hbm | 16x16x64/s1 | 24.8 | 1561 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 1024 | 4096 | 4096 | 116.90 | 92.55 | 0.792 | compute | 104.12 | 0.891 | compute:tiles | 128x256x16/s1 | 293.9 | 431 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 1 | 11008 | 4096 | 55.93 | 44.67 | 0.799 | memory | 44.78 | 0.801 | memory:hbm | 16x16x16/s1 | 1.6 | 1613 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 2 | 11008 | 4096 | 55.94 | 44.68 | 0.799 | memory | 44.79 | 0.801 | memory:hbm | 16x16x16/s1 | 3.2 | 1613 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8 | 11008 | 4096 | 55.99 | 44.77 | 0.800 | memory | 44.89 | 0.802 | memory:hbm | 16x16x16/s1 | 12.9 | 1615 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 16 | 11008 | 4096 | 55.59 | 44.89 | 0.808 | memory | 45.02 | 0.810 | memory:hbm | 16x16x16/s1 | 26.0 | 1631 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 1 | 4096 | 11008 | 53.74 | 44.67 | 0.831 | memory | 44.78 | 0.833 | memory:hbm | 16x16x64/s1 | 1.7 | 1679 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 2 | 4096 | 11008 | 53.61 | 44.68 | 0.834 | memory | 44.79 | 0.836 | memory:hbm | 16x16x64/s1 | 3.4 | 1683 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 32 | 24576 | 6144 | 173.13 | 150.51 | 0.869 | memory | 150.96 | 0.872 | memory:hbm | 16x64x16/s1 | 55.8 | 1756 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 32 | 6144 | 12288 | 86.20 | 75.35 | 0.874 | memory | 75.55 | 0.877 | memory:hbm | 16x64x16/s1 | 56.1 | 1765 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 1 | 6144 | 12288 | 80.06 | 74.79 | 0.934 | memory | 74.97 | 0.936 | memory:hbm | 16x16x32/s1 | 1.9 | 1886 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 2 | 6144 | 12288 | 79.94 | 74.80 | 0.936 | memory | 74.98 | 0.938 | memory:hbm | 16x16x32/s1 | 3.8 | 1890 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8 | 6144 | 12288 | 79.92 | 74.91 | 0.937 | memory | 75.10 | 0.940 | memory:hbm | 16x16x32/s1 | 15.1 | 1893 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 16 | 6144 | 12288 | 79.68 | 75.06 | 0.942 | memory | 75.25 | 0.944 | memory:hbm | 16x16x32/s1 | 30.3 | 1903 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 16 | 24576 | 6144 | 156.68 | 150.02 | 0.958 | memory | 150.43 | 0.960 | memory:hbm | 16x16x16/s1 | 30.8 | 1934 | Cijk_Ailk_Bljk_BBH_MT384x16x32_SE_AMAS3_ x1 |
| 8 | 24576 | 6144 | 156.39 | 149.78 | 0.958 | memory | 150.16 | 0.960 | memory:hbm | 16x16x16/s1 | 15.4 | 1934 | Cijk_Ailk_Bljk_BBH_MT384x16x32_SE_AMAS3_ x1 |
| 2 | 24576 | 6144 | 156.10 | 149.60 | 0.958 | memory | 149.96 | 0.961 | memory:hbm | 16x16x16/s1 | 3.9 | 1935 | Cijk_Ailk_Bljk_BBH_MT384x16x32_SE_AMAS3_ x1 |
| 1 | 24576 | 6144 | 155.99 | 149.57 | 0.959 | memory | 149.93 | 0.961 | memory:hbm | 16x16x16/s1 | 1.9 | 1936 | Cijk_Ailk_Bljk_BBH_MT384x16x32_SE_AMAS3_ x1 |
