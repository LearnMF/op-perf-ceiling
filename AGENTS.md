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
- Both nodes go through `bw-zz`; reuse the user's ControlMaster (`ssh -O check bw-zz`), never type passwords.
- gfx938 / BW1102: node `nmz5`, container `tanbo_kda_nmz` (stopped by default: `docker start` it, `docker stop` it afterwards) (DTK 26.10, torch 2.11), home mounted at `/public4/home/tanbo`. Check `hy-smi` for an idle card first; run `bench/gemm_sweep.py` with `HIP_VISIBLE_DEVICES=<idle card>` inside the container after `source /opt/dtk/env.sh`.
- gfx936 / BW1000: node `bw52` (ssh via `bw-zz`), running container `tanbo_mg2610` (DTK 26.10, torch 2.11), home `/public/home/tanbo`. (`bw7` from the old repo is gone.)
- Committed sweeps go to `data/measured/gemm_<dtype>_<machine id>_<date>.jsonl`; `tests/test_check.py` picks them up automatically.
