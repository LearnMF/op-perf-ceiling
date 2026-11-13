# opceil check: GEMM T_bound vs kernel-level measurement

machine file `gfx936-bw1000` (measured peaks: mmac_bf16 476 TFLOPS, HBM 1366 GB/s); measured on bw52 BW1000 64G (80 CUs), torch 2.11.0, 2026-11-13 17:59:00

| | value |
| --- | ---: |
| shapes | 104 |
| **bound violations** (T_bound > 1.01 x measured) | **0** |
| eta = T_bound / measured: p10 / median / p90 | 0.177 / 0.537 / 0.684 |
| Spearman(T_bound, measured) | 0.972 |
| T_split (tiled, model) above measured (> 1.01 x) | 0 |
| eta_split = T_split / measured: p10 / median / p90 | 0.215 / 0.693 / 0.924 |
| Spearman(T_split, measured) | 0.966 |

eta = 1 means the measured kernel runs at the bound; eta = 0.5 means it takes twice the bound.  T_split above measured means the tiled model claims more cost than the vendor BLAS actually pays: the model is wrong there, not the library.

## by bottleneck side of the bound

| side | n | p10 | median | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| compute | 18 | 0.249 | 0.443 | 0.601 | 0.660 |
| fixed | 7 | 0.146 | 0.183 | 0.186 | 0.300 |
| memory | 79 | 0.180 | 0.595 | 0.686 | 0.741 |

## by size (T_bound | T_split)

| min(M,N) | n | eta p10 | eta median | eta p90 | eta_split p10 | eta_split median | eta_split p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| min(M,N) <= 16 | 40 | 0.183 | 0.624 | 0.686 | 0.208 | 0.843 | 0.926 |
| 17-256 | 44 | 0.155 | 0.451 | 0.655 | 0.215 | 0.618 | 0.885 |
| 257-2048 | 18 | 0.210 | 0.428 | 0.594 | 0.279 | 0.478 | 0.670 |
| > 2048 | 2 | 0.601 | 0.604 | 0.606 | 0.690 | 0.693 | 0.696 |

## all shapes (sorted by eta)

| M | N | K | measured us | T_bound us | eta | side | T_split us | eta_split | split binding | tile | TFLOPS | GB/s | kernels |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 1 | 256 | 6144 | 21.97 | 2.18 | 0.099 | fixed | 2.31 | 0.105 | memory:hbm | 16x16x16/s32 | 0.1 | 144 | Cijk_Ailk_Bljk_BBH_MT32x32x128_SE_AMAS3_ x1 |
| 256 | 256 | 6144 | 29.31 | 3.48 | 0.119 | memory | 4.96 | 0.169 | memory:hbm | 64x64x16/s8 | 27.5 | 219 | Cijk_Ailk_Bljk_BBH_MT32x32x128_SE_AMAS3_ x1 |
| 64 | 2304 | 1152 | 23.37 | 3.12 | 0.133 | memory | 4.26 | 0.182 | memory:hbm | 32x32x32/s1 | 14.5 | 246 | Cijk_Ailk_Bljk_BBH_MT32x16x64_SE_AMAS3_B x1 |
| 128 | 256 | 6144 | 18.18 | 2.60 | 0.143 | memory | 3.63 | 0.200 | memory:hbm | 32x64x16/s16 | 22.2 | 263 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS0_ x1 |
| 64 | 256 | 6144 | 14.95 | 2.18 | 0.146 | fixed | 2.97 | 0.199 | memory:hbm | 16x64x16/s16 | 13.5 | 265 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS0_ x1 |
| 1 | 576 | 4608 | 18.83 | 2.89 | 0.153 | memory | 3.90 | 0.207 | memory:hbm | 16x16x16/s16 | 0.3 | 282 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 256 | 2304 | 1152 | 24.80 | 3.84 | 0.155 | memory | 5.41 | 0.218 | memory:l2 | 64x128x32/s1 | 54.8 | 285 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 16 | 576 | 4608 | 18.97 | 2.97 | 0.157 | memory | 4.05 | 0.213 | memory:hbm | 16x16x16/s16 | 4.5 | 289 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 2 | 576 | 4608 | 17.69 | 2.89 | 0.163 | memory | 3.91 | 0.221 | memory:hbm | 16x16x16/s16 | 0.6 | 301 | Cijk_Ailk_Bljk_BBH_MT32x32x128_SE_AMAS3_ x1 |
| 512 | 576 | 4608 | 34.29 | 5.76 | 0.168 | memory | 8.95 | 0.261 | memory:hbm | 128x128x16/s4 | 79.3 | 310 | Cijk_Ailk_Bljk_BBH_MT64x64x64_SE_AMAS3_B x1 |
| 32 | 256 | 6144 | 12.29 | 2.18 | 0.177 | fixed | 2.64 | 0.215 | memory:hbm | 16x16x16/s16 | 8.2 | 289 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 64 | 576 | 4608 | 17.97 | 3.24 | 0.180 | memory | 4.52 | 0.252 | memory:hbm | 16x64x32/s4 | 18.9 | 332 | Cijk_Ailk_Bljk_BBH_MT32x32x128_SE_AMAS3_ x1 |
| 16 | 256 | 6144 | 11.90 | 2.18 | 0.183 | fixed | 2.47 | 0.208 | memory:hbm | 16x16x16/s32 | 4.2 | 282 | Cijk_Ailk_Bljk_BBH_MT32x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 8 | 256 | 6144 | 11.82 | 2.18 | 0.184 | fixed | 2.39 | 0.202 | memory:hbm | 16x16x16/s32 | 2.1 | 275 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 128 | 576 | 4608 | 19.40 | 3.60 | 0.186 | memory | 5.15 | 0.266 | memory:hbm | 32x128x16/s8 | 35.0 | 342 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 2 | 256 | 6144 | 11.72 | 2.18 | 0.186 | fixed | 2.32 | 0.198 | memory:hbm | 16x16x16/s32 | 0.5 | 271 | Cijk_Ailk_Bljk_BBH_MT64x16x32_SE_AMAS3_B x1, Cijk_B_PostGSU x1 |
| 1024 | 1024 | 1024 | 23.32 | 4.37 | 0.187 | compute | 5.84 | 0.251 | memory:l2 | 128x128x32/s1 | 92.1 | 270 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 512 | 256 | 6144 | 27.27 | 5.26 | 0.193 | memory | 7.63 | 0.280 | memory:hbm | 64x128x16/s8 | 59.1 | 356 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 8 | 2304 | 1152 | 14.13 | 2.91 | 0.206 | memory | 3.93 | 0.278 | memory:hbm | 16x16x64/s1 | 3.0 | 380 | Cijk_Ailk_Bljk_BBH_MT64x16x64_SE_AMAS3_B x1 |
| 256 | 576 | 4608 | 20.73 | 4.32 | 0.208 | memory | 6.42 | 0.310 | memory:hbm | 64x128x16/s8 | 65.5 | 384 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 512 | 2304 | 1152 | 26.35 | 5.53 | 0.210 | compute | 7.35 | 0.279 | memory:l2 | 128x128x16/s1 | 103.1 | 336 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 8 | 576 | 4608 | 12.91 | 2.92 | 0.227 | memory | 3.97 | 0.307 | memory:hbm | 16x16x16/s16 | 3.3 | 418 | Cijk_Ailk_Bljk_BBH_MT16x16x128_SE_AMAS3_ x1 |
| 32 | 2304 | 1152 | 12.79 | 3.00 | 0.235 | memory | 4.08 | 0.319 | memory:hbm | 16x16x32/s1 | 13.3 | 432 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 2 | 2304 | 1152 | 12.09 | 2.89 | 0.239 | memory | 3.90 | 0.323 | memory:hbm | 16x16x64/s1 | 0.9 | 440 | Cijk_Ailk_Bljk_BBH_MT16x16x128_SE_AMAS3_ x1 |
| 1 | 2304 | 1152 | 11.77 | 2.88 | 0.245 | memory | 3.89 | 0.331 | memory:hbm | 16x16x64/s1 | 0.5 | 452 | Cijk_Ailk_Bljk_BBH_MT32x16x64_SE_AMAS3_B x1 |
| 128 | 2304 | 1152 | 13.53 | 3.36 | 0.248 | memory | 4.66 | 0.344 | memory:hbm | 32x128x32/s1 | 50.2 | 458 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1 |
| 1024 | 576 | 4608 | 44.47 | 11.06 | 0.249 | compute | 14.01 | 0.315 | memory:hbm | 256x128x16/s4 | 122.2 | 358 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 1024 | 256 | 6144 | 35.16 | 8.82 | 0.251 | memory | 12.95 | 0.368 | memory:hbm | 64x256x16/s8 | 91.6 | 462 | Cijk_Ailk_Bljk_BBH_MT64x64x64_SE_AMAS3_B x1 |
| 32 | 576 | 4608 | 12.05 | 3.06 | 0.254 | memory | 4.16 | 0.345 | memory:hbm | 16x16x128/s1 | 14.1 | 468 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 512 | 512 | 512 | 7.25 | 2.18 | 0.300 | fixed | 2.18 | 0.300 | fixed:launch | 32x64x32/s1 | 37.0 | 217 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1 |
| 16 | 2304 | 1152 | 9.66 | 2.94 | 0.304 | memory | 3.98 | 0.412 | memory:hbm | 16x16x64/s1 | 8.8 | 561 | Cijk_Ailk_Bljk_BBH_MT32x16x128_SE_AMAS3_ x1 |
| 1024 | 2304 | 1152 | 35.70 | 11.06 | 0.310 | compute | 12.69 | 0.355 | compute:mmac_bf16 | 128x256x16/s1 | 152.3 | 347 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 2048 | 256 | 6144 | 50.54 | 15.93 | 0.315 | memory | 23.59 | 0.467 | memory:hbm | 64x256x16/s4 | 127.5 | 581 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 2048 | 576 | 4608 | 61.50 | 22.12 | 0.360 | compute | 25.38 | 0.413 | compute:tiles | 256x128x16/s2 | 176.8 | 432 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 256 | 6144 | 12288 | 236.06 | 87.04 | 0.369 | memory | 123.75 | 0.524 | memory:hbm | 128x128x16/s2 | 163.7 | 680 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 6144 | 2048 | 42.87 | 15.93 | 0.372 | memory | 22.56 | 0.526 | compute:tiles | 64x256x16/s1 | 150.3 | 685 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 4096 | 576 | 4608 | 115.40 | 44.24 | 0.383 | compute | 50.76 | 0.440 | compute:tiles | 256x128x16/s1 | 188.4 | 414 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 8192 | 256 | 6144 | 149.97 | 58.60 | 0.391 | memory | 79.83 | 0.532 | memory:hbm | 128x128x16/s1 | 171.8 | 720 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 4096 | 6144 | 72.79 | 30.15 | 0.414 | memory | 41.44 | 0.569 | memory:hbm | 128x128x16/s1 | 177.0 | 763 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 4096 | 256 | 6144 | 72.46 | 30.15 | 0.416 | memory | 41.44 | 0.572 | memory:hbm | 128x128x16/s1 | 177.8 | 767 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 512 | 4096 | 11008 | 220.26 | 93.93 | 0.426 | compute | 103.81 | 0.471 | memory:l2 | 256x256x16/s32 | 209.6 | 480 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 512 | 4096 | 6144 | 122.20 | 52.43 | 0.429 | compute | 59.22 | 0.485 | compute:tiles | 256x256x16/s16 | 210.9 | 498 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 4096 | 11008 | 122.49 | 53.12 | 0.434 | memory | 72.74 | 0.594 | memory:hbm | 128x128x16/s1 | 188.5 | 799 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 512 | 4096 | 4096 | 80.40 | 34.95 | 0.435 | compute | 44.69 | 0.556 | memory:l2 | 128x256x16/s8 | 213.7 | 522 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 256 | 4096 | 4096 | 46.71 | 20.48 | 0.438 | memory | 28.26 | 0.605 | memory:hbm | 64x256x16/s1 | 183.9 | 808 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 512 | 11008 | 4096 | 208.15 | 93.93 | 0.451 | compute | 101.52 | 0.488 | compute:tiles | 256x256x16/s8 | 221.8 | 508 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 512 | 6144 | 2048 | 57.13 | 26.21 | 0.459 | compute | 34.03 | 0.596 | memory:l2 | 128x128x16/s1 | 225.6 | 587 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 8192 | 576 | 4608 | 192.23 | 88.47 | 0.460 | compute | 101.52 | 0.528 | compute:tiles | 256x128x16/s1 | 226.2 | 469 | Cijk_Ailk_Bljk_BBH_MT256x384x16_SE_AMAS3 x1 |
| 64 | 6144 | 2048 | 30.64 | 14.22 | 0.464 | memory | 19.33 | 0.631 | memory:hbm | 16x128x16/s1 | 52.6 | 856 | Cijk_Ailk_Bljk_BBH_MT64x64x32_SE_AMAS3_B x1 |
| 256 | 11008 | 4096 | 104.03 | 53.12 | 0.511 | memory | 72.69 | 0.699 | memory:hbm | 128x128x16/s1 | 221.9 | 941 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 128 | 6144 | 2048 | 28.28 | 14.79 | 0.523 | memory | 20.24 | 0.716 | memory:hbm | 32x256x16/s1 | 113.9 | 964 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 128 | 4096 | 4096 | 36.16 | 19.34 | 0.535 | memory | 26.29 | 0.727 | memory:hbm | 64x64x16/s1 | 118.8 | 986 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 128 | 4096 | 6144 | 53.37 | 28.73 | 0.538 | memory | 38.96 | 0.730 | memory:hbm | 64x64x16/s1 | 120.7 | 992 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 128 | 4096 | 11008 | 92.15 | 51.02 | 0.554 | memory | 69.05 | 0.749 | memory:hbm | 64x64x16/s1 | 125.3 | 1021 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 32 | 6144 | 2048 | 24.63 | 13.94 | 0.566 | memory | 18.88 | 0.767 | memory:hbm | 16x16x16/s1 | 32.7 | 1043 | Cijk_Ailk_Bljk_BBH_MT128x32x64_SE_AMAS3_ x1 |
| 64 | 4096 | 6144 | 48.73 | 28.02 | 0.575 | memory | 37.91 | 0.778 | memory:hbm | 16x128x16/s1 | 66.1 | 1060 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 1 | 6144 | 2048 | 23.44 | 13.66 | 0.583 | memory | 18.44 | 0.787 | memory:hbm | 16x16x16/s1 | 1.1 | 1074 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS3_ x1 |
| 2048 | 2304 | 1152 | 37.89 | 22.12 | 0.584 | compute | 25.38 | 0.670 | compute:tiles | 128x256x16/s1 | 286.9 | 514 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 64 | 4096 | 4096 | 31.71 | 18.77 | 0.592 | memory | 25.43 | 0.802 | memory:hbm | 16x128x16/s1 | 67.7 | 1091 | Cijk_Ailk_Bljk_BBH_MT64x64x64_SE_AMAS3_B x1 |
| 1024 | 6144 | 2048 | 88.33 | 52.43 | 0.594 | compute | 56.40 | 0.638 | compute:tiles | 256x256x16/s4 | 291.7 | 475 | Cijk_Ailk_Bljk_BBH_MT256x384x16_SE_AMAS3 x1 |
| 8 | 6144 | 2048 | 23.08 | 13.72 | 0.595 | memory | 18.54 | 0.803 | memory:hbm | 16x16x16/s1 | 8.7 | 1096 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS3_ x1 |
| 128 | 11008 | 4096 | 85.18 | 51.02 | 0.599 | memory | 69.36 | 0.814 | memory:hbm | 32x256x16/s1 | 135.5 | 1104 | Cijk_Ailk_Bljk_BBH_MT256x128x16_SE_AMAS3 x1 |
| 2 | 6144 | 2048 | 22.80 | 13.67 | 0.600 | memory | 18.46 | 0.809 | memory:hbm | 16x16x16/s1 | 2.2 | 1105 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS3_ x1 |
| 2048 | 2048 | 2048 | 58.16 | 34.95 | 0.601 | compute | 45.12 | 0.776 | compute:tiles | 128x256x16/s1 | 295.4 | 433 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 4096 | 2304 | 1152 | 73.60 | 44.24 | 0.601 | compute | 50.76 | 0.690 | compute:tiles | 128x256x16/s1 | 295.4 | 457 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 16 | 4096 | 4096 | 30.44 | 18.35 | 0.603 | memory | 24.79 | 0.814 | memory:hbm | 16x16x32/s1 | 17.6 | 1111 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 16 | 6144 | 2048 | 22.88 | 13.80 | 0.603 | memory | 18.66 | 0.815 | memory:hbm | 16x16x16/s1 | 17.6 | 1111 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8192 | 2304 | 1152 | 145.90 | 88.47 | 0.606 | compute | 101.52 | 0.696 | compute:tiles | 128x256x16/s1 | 298.1 | 424 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 64 | 11008 | 4096 | 81.30 | 49.97 | 0.615 | memory | 67.70 | 0.833 | memory:hbm | 16x128x16/s1 | 71.0 | 1133 | Cijk_Ailk_Bljk_BBH_MT128x128x16_SE_AMAS3 x1 |
| 32 | 4096 | 4096 | 30.05 | 18.49 | 0.615 | memory | 25.00 | 0.832 | memory:hbm | 16x16x16/s1 | 35.7 | 1134 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 8 | 4096 | 4096 | 29.66 | 18.28 | 0.616 | memory | 24.68 | 0.832 | memory:hbm | 16x16x32/s1 | 9.0 | 1136 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 32 | 4096 | 6144 | 44.79 | 27.66 | 0.618 | memory | 37.38 | 0.835 | memory:hbm | 16x16x16/s1 | 36.0 | 1138 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS0_ x1 |
| 16 | 4096 | 6144 | 44.26 | 27.48 | 0.621 | memory | 37.12 | 0.839 | memory:hbm | 16x16x32/s1 | 18.2 | 1145 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 32 | 11008 | 4096 | 79.47 | 49.45 | 0.622 | memory | 66.87 | 0.841 | memory:hbm | 16x16x16/s1 | 36.3 | 1147 | Cijk_Ailk_Bljk_BBH_MT128x32x32_SE_AMAS3_ x1 |
| 2 | 4096 | 4096 | 29.24 | 18.22 | 0.623 | memory | 24.60 | 0.841 | memory:hbm | 16x16x32/s1 | 2.3 | 1149 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 1 | 4096 | 4096 | 29.12 | 18.21 | 0.626 | memory | 24.58 | 0.844 | memory:hbm | 16x16x32/s1 | 1.2 | 1153 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 8 | 4096 | 6144 | 43.12 | 27.40 | 0.635 | memory | 36.99 | 0.858 | memory:hbm | 16x16x32/s1 | 9.3 | 1171 | Cijk_Ailk_Bljk_BBH_MT64x16x128_SE_AMAS3_ x1 |
| 128 | 24576 | 6144 | 263.74 | 168.11 | 0.637 | memory | 228.01 | 0.865 | memory:hbm | 32x256x16/s1 | 146.6 | 1175 | Cijk_Ailk_Bljk_BBH_MT384x128x32_SE_AMAS3 x1 |
| 128 | 6144 | 12288 | 132.33 | 84.48 | 0.638 | memory | 114.30 | 0.864 | memory:hbm | 32x256x16/s1 | 146.0 | 1177 | Cijk_Ailk_Bljk_BBH_MT128x128x32_SE_AMAS3 x1 |
| 64 | 4096 | 11008 | 78.21 | 49.97 | 0.639 | memory | 67.54 | 0.864 | memory:hbm | 16x128x16/s1 | 73.8 | 1178 | Cijk_Ailk_Bljk_BBH_MT256x64x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 2 | 24576 | 6144 | 251.09 | 163.91 | 0.653 | memory | 221.25 | 0.881 | memory:hbm | 16x16x16/s1 | 2.4 | 1203 | Cijk_Ailk_Bljk_BBH_MT256x16x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 1 | 4096 | 6144 | 41.77 | 27.32 | 0.654 | memory | 36.87 | 0.883 | memory:hbm | 16x16x32/s1 | 1.2 | 1206 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 32 | 4096 | 11008 | 75.51 | 49.45 | 0.655 | memory | 66.79 | 0.885 | memory:hbm | 16x16x16/s1 | 38.2 | 1207 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS0_ x1 |
| 1 | 24576 | 6144 | 249.86 | 163.87 | 0.656 | memory | 221.19 | 0.885 | memory:hbm | 16x16x16/s1 | 1.2 | 1209 | Cijk_Ailk_Bljk_BBH_MT256x16x32_SE_AMAS3_ x1, Cijk_B_PostGSU x1 |
| 2 | 4096 | 6144 | 41.64 | 27.33 | 0.656 | memory | 36.89 | 0.886 | memory:hbm | 16x16x32/s1 | 2.4 | 1210 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 8 | 4096 | 11008 | 74.73 | 49.06 | 0.656 | memory | 66.22 | 0.886 | memory:hbm | 16x16x32/s1 | 9.7 | 1210 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 8 | 24576 | 6144 | 248.67 | 164.11 | 0.660 | memory | 221.57 | 0.891 | memory:hbm | 16x16x16/s1 | 9.7 | 1216 | Cijk_Ailk_Bljk_BBH_MT320x64x32_SE_AMAS3_ x1 |
| 1024 | 4096 | 4096 | 105.86 | 69.91 | 0.660 | compute | 78.96 | 0.746 | compute:tiles | 128x256x16/s4 | 324.6 | 475 | Cijk_Ailk_Bljk_BBH_MT256x256x16_SE_AMAS3 x1 |
| 16 | 4096 | 11008 | 74.41 | 49.19 | 0.661 | memory | 66.41 | 0.893 | memory:hbm | 16x16x32/s1 | 19.4 | 1218 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS0_ x1 |
| 2 | 4096 | 11008 | 72.87 | 48.96 | 0.672 | memory | 66.08 | 0.907 | memory:hbm | 16x16x32/s1 | 2.5 | 1238 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 1 | 4096 | 11008 | 72.60 | 48.94 | 0.674 | memory | 66.06 | 0.910 | memory:hbm | 16x16x32/s1 | 1.2 | 1243 | Cijk_Ailk_Bljk_BBH_MT64x32x64_SE_AMAS3_B x1 |
| 16 | 24576 | 6144 | 242.90 | 164.37 | 0.677 | memory | 222.00 | 0.914 | memory:hbm | 16x16x16/s1 | 19.9 | 1247 | Cijk_Ailk_Bljk_BBH_MT320x64x32_SE_AMAS3_ x1 |
| 2 | 6144 | 12288 | 120.06 | 81.96 | 0.683 | memory | 110.63 | 0.921 | memory:hbm | 16x16x16/s1 | 2.5 | 1258 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS0_ x1 |
| 1 | 6144 | 12288 | 119.71 | 81.94 | 0.684 | memory | 110.60 | 0.924 | memory:hbm | 16x16x16/s1 | 1.3 | 1262 | Cijk_Ailk_Bljk_BBH_MT128x16x64_SE_AMAS0_ x1 |
| 2 | 11008 | 4096 | 71.45 | 48.96 | 0.685 | memory | 66.09 | 0.925 | memory:hbm | 16x16x16/s1 | 2.5 | 1263 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 1 | 11008 | 4096 | 71.33 | 48.94 | 0.686 | memory | 66.06 | 0.926 | memory:hbm | 16x16x16/s1 | 1.3 | 1265 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8 | 11008 | 4096 | 71.35 | 49.06 | 0.688 | memory | 66.24 | 0.928 | memory:hbm | 16x16x16/s1 | 10.1 | 1267 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 8 | 6144 | 12288 | 119.17 | 82.08 | 0.689 | memory | 110.80 | 0.930 | memory:hbm | 16x16x16/s1 | 10.1 | 1269 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 16 | 11008 | 4096 | 70.46 | 49.19 | 0.698 | memory | 66.45 | 0.943 | memory:hbm | 16x16x16/s1 | 20.5 | 1287 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 16 | 6144 | 12288 | 117.54 | 82.24 | 0.700 | memory | 111.04 | 0.945 | memory:hbm | 16x16x16/s1 | 20.6 | 1290 | Cijk_Ailk_Bljk_BBH_MT128x16x32_SE_AMAS3_ x1 |
| 32 | 6144 | 12288 | 117.62 | 82.56 | 0.702 | memory | 111.50 | 0.948 | memory:hbm | 16x16x16/s1 | 41.1 | 1294 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 64 | 6144 | 12288 | 115.66 | 83.20 | 0.719 | memory | 112.44 | 0.972 | memory:hbm | 16x128x16/s1 | 83.6 | 1326 | Cijk_Ailk_Bljk_BBH_MT128x64x32_SE_AMAS3_ x1 |
| 64 | 24576 | 6144 | 225.24 | 165.97 | 0.737 | memory | 224.58 | 0.997 | memory:hbm | 16x128x16/s1 | 85.8 | 1358 | Cijk_Ailk_Bljk_BBH_MT320x64x32_SE_AMAS3_ x1 |
| 32 | 24576 | 6144 | 222.58 | 164.91 | 0.741 | memory | 222.86 | 1.001 | memory:hbm | 16x16x16/s1 | 43.4 | 1366 | Cijk_Ailk_Bljk_BBH_MT320x64x32_SE_AMAS3_ x1 |
