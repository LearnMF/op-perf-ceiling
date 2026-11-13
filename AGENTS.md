# AGENTS.md

## Environment and checks
- Python env: `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e ".[dev]"`; add `torch==2.7.1 numpy` for the torch front end (tests skip without it).
- Tests: `.venv/bin/python -m pytest -q` (about 80 s; the KDA split searches dominate). Must stay green.
- CLI: `.venv/bin/python -m opceil {predict,trace,check,validate,calibrate}`.

## Rules of the model (do not break)
- Every term in `opceil/model/predict.py:_terms` carries a class: `bound` / `model` / `calibrated`. A new term must set its class; anything resting on an implementation assumption is `model`, anything using `structural.*` is `calibrated`. `strict=True` must keep only `bound` terms. `T_bound <= T_theory <= T_split` is tested.
- No constant fitted to, or lifted from, an optimised kernel's measurements goes into a prediction. New heuristic constants are registered in `docs/devlog/constant-ledger.md`.
- A missing machine-file field is never guessed: the term is dropped and a note says so. A theoretical peak below a measured one is a data error (`tests/test_check.py`).
- Operators without a plugin are refused, not estimated. Plugin authoring: `docs/plugin-guide.md`.

## Real-card measurements
- gfx938 / BW1102: node `nmz5` (ssh via `bw-zz`, needs the user's ControlMaster: `ssh -O check bw-zz`), container `tanbo_kda_nmz` (DTK 26.10, torch 2.11), home mounted at `/public4/home/tanbo`. Check `hy-smi` for an idle card first; run `bench/gemm_sweep.py` with `HIP_VISIBLE_DEVICES=<idle card>` inside the container after `source /opt/dtk/env.sh`.
- gfx936 / BW1000: historically `bw7` (container `tanbo_mega_k3`), not reachable as of 2026-11-13 (hostname not resolvable from bw-zz; 10.16.1.47 rejects the key). Ask the user for a node.
- Committed sweeps go to `data/measured/gemm_<dtype>_<machine id>_<date>.jsonl`; `tests/test_check.py` picks them up automatically.
