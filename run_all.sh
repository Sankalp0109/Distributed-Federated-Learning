#!/usr/bin/env bash
# Full experiment suite (~25 min on an 8+ core laptop CPU). Results land in results/<name>/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
R=${ROUNDS:-30}
run() { $PY run_experiment.py --rounds "$R" "$@"; }

$PY -m fl.centralized --epochs "$R" --out results/centralized

# Data heterogeneity
run --name iid        --split iid
run --name shards     --split shards --classes-per-worker 2
run --name dirichlet  --split dirichlet --alpha 0.3
run --name mixed      --split mixed --n-skewed 2 --classes-per-worker 2

# Communication / computation trade-offs
run --name iid_fp16   --split iid --wire-dtype float16
run --name iid_frac60 --split iid --clients-fraction 0.6 --min-clients 2
run --name iid_e5     --split iid --local-epochs 5 --round-timeout 30
run --name shards_e5  --split shards --local-epochs 5 --round-timeout 30

# Stragglers: worker 4 is always slower than the 10 s round timeout
run --name straggler_wait   --slow 4:12:2 --min-clients 3
run --name straggler_quorum --slow 4:12:2 --min-clients 4 --early-quorum

# Crashes / silent failures
run --name crash         --crash 3:10 --crash-recover 4:10:20 --min-clients 3
run --name hang_drop     --hang 2:10 --drop 1:0.3 --min-clients 3
run --name strict_quorum --crash 3:10 --min-clients 5
run --name crash_shards  --split shards --crash 0:10 --min-clients 3

$PY plot_results.py
