# Unknown hardware from anchor kernels (PV SOC simulation: S4000 / S2000)

## 1. GEMM, anchors on the target card only (1 floor + 1 memory + 1 compute row -> L0 roofline for the rest)

| card | anchor combos | median error (median over combos) | worst combo median | typical max error |
| --- | ---: | ---: | ---: | ---: |
| s4000 | 24 | 11.4% | 30.0% | 39.9% |
| s2000 | 24 | 10.9% | 14.1% | 39.4% |

## 2. S2000 -> S4000 for other families

ratios S4000/S2000 from the GEMM anchors: floor anchor 1.007, memory anchor 0.602, compute anchor 0.459, whole GEMM median 0.621, 'S2200 = 83% x S2000' style: one flat 1/1.0 1.000

### DSA decode (14 configs, measured ratio median 0.793)

| ratio used | median | mean | max | within 10% |
| --- | ---: | ---: | ---: | ---: |
| floor anchor | 27.2% | 32.2% | 69.6% | 0 / 14 |
| memory anchor | 23.9% | 21.1% | 33.0% | 2 / 14 |
| compute anchor | 42.0% | 39.7% | 48.9% | 0 / 14 |
| whole GEMM median | 21.5% | 19.5% | 30.9% | 3 / 14 |
| 'S2200 = 83% x S2000' style: one flat 1/1.0 | 26.3% | 31.3% | 68.4% | 0 / 14 |

### MoE (tokens [16, 64, 2048, 4096, 8192]; EP {'s2000': 16, 's4000': 8})

| ratio used | median | mean | max | within 10% |
| --- | ---: | ---: | ---: | ---: |
| floor anchor | 26.6% | 28.3% | 42.7% | 0 / 5 |
| memory anchor | 24.3% | 23.3% | 30.1% | 0 / 5 |
| compute anchor | 42.3% | 41.5% | 46.7% | 0 / 5 |
| whole GEMM median | 21.9% | 20.8% | 27.9% | 0 / 5 |
| 'S2200 = 83% x S2000' style: one flat 1/1.0 | 25.7% | 27.4% | 41.7% | 0 / 5 |
| L0 model ratio on the anchor machines, each card's own EP | 51.4% | 50.2% | 62.0% | 0 / 5 |
| L0 model alone on S4000's anchor machine (no S2000 data) | 58.2% | 69.4% | 124.1% | 0 / 5 |
