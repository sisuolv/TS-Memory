# TS‑Memory: ChronosBolt + TS‑Memory (Quantile)

This repository packages a **from-scratch** pipeline for running **TS‑Memory (quantile)** on top of a **ChronosBolt (base)** foundation model:

1) Build **retrieval_database** (ChronosBolt embeddings) from raw CSV  
2) Build offline **teacher_ds** (retrieval‑distilled quantile targets)  
3) (Recommended) Cache **q_base** into teacher shards  
4) Train **Memory** (one checkpoint per `pred_len`)  
5) Evaluate **NO‑RETRIEVAL** inference with **automatic alpha selection** (default: fixed `[0,1]` grid at step `0.05`)

All outputs are written to log files; no result post‑processing scripts are required.

## Quickstart

Run everything from `src/`:

```bash
cd src
```

### Data

Set `DATA_ROOT` to a folder that contains datasets in the Time‑Series‑Library style, e.g.:

```
<DATA_ROOT>/
  ETT-small/ETTh1.csv ...
  weather/weather.csv
  traffic/traffic.csv
  electricity/electricity.csv
  exchange_rate/exchange_rate.csv
```

If unset, scripts try to auto-detect `../all_datasets` or `../../all_datasets` (relative to `src/`).

### Model

Set `BASE_MODEL_PATH` to a ChronosBolt checkpoint directory containing `config.json` and weights.  
If unset, scripts default to `checkpoints/base` (relative to `src/`).

## Run (Local)

Example (single dataset, multiple horizons):

```bash
cd src
DATASET_NAME=ETTh1 PRED_LENS="96 192" bash script/run_maer_v2_local.sh
```

Logs: `src/logs/local/`  
Artifacts (gitignored): `src/retrieval_database/`, `src/teacher_ds/`, `src/checkpoints/`, `src/results/`

## Run (Slurm)

End‑to‑end fullgrid submit script (build retrdb → teacher → q_base cache → train → eval):

```bash
cd src
bash script/submit_maer_v2_final_fullgrid.sh
```

By default, eval searches `alpha` on a fixed grid and selects the best value on the **test** split (oracle / leaky).
To select on the **val** split instead (research‑valid):

```bash
EVAL_ALPHA_POLICY=val_auto bash script/submit_maer_v2_final_fullgrid.sh
```

Common overrides:

```bash
DATASETS="ETTh1 ETTh2" PRED_LENS="64 96" RETRDB_FEATURE_SHARD_TOTAL=8 bash script/submit_maer_v2_final_fullgrid.sh
```

Slurm logs: `src/logs/%x-%j.out` and `src/logs/%x-%j.err`.
