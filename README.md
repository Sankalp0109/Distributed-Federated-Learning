# Distributed Federated Learning

A coordinator and 4–6 worker processes train a small CNN on MNIST using
sample-weighted FedAvg over TCP. Each worker holds its own private data shard.
Every round has a round id and a deadline. Late updates are rejected, and a
round only aggregates if a stated minimum number of workers reported in time.

## Layout

```
fl/protocol.py     framed TCP messages (4-byte length + pickle), exact byte counters
fl/model.py        SmallCNN (46,730 params), local SGD, evaluation, FedAvg
fl/data.py         MNIST / Fashion-MNIST loading; iid / shards / dirichlet / mixed partitions
fl/coordinator.py  registration, round loop, timeouts, stale rejection, min participation, metrics
fl/worker.py       local training + fault injection (slow / crash / crash-recover / hang / drop)
fl/centralized.py  centralized baseline with the same model and optimizer
run_experiment.py  launches 1 coordinator + N worker processes for one configuration
run_all.sh         the full experiment suite
plot_results.py    figures (results/figures/) + summary table (results/summary.md)
```

## Setup and running

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision numpy pandas matplotlib tabulate
.venv/bin/python run_experiment.py --name iid --split iid --rounds 30     # one run
./run_all.sh                                                              # full suite + plots
```

MNIST downloads to `data/` on first use (~12 MB). Each run writes these files to `results/<name>/`:
`config.json`, `partition.json` (per-worker label histograms), `metrics.jsonl`
(one line per round), `coordinator.log`, and `worker_<i>.log`.

## Protocol and round semantics

```
worker                         coordinator
  |-- REGISTER(id, n_samples) -->|   (also used to re-join after a crash)
  |<-------- REGISTER_ACK -------|
  |<-- TRAIN(round r, weights) --|   broadcast to selected live workers, deadline = now + T
  |  local SGD for E epochs      |
  |--- UPDATE(round r, w, n) --->|   accepted only if r == current round and before deadline
  |<----- REJECT(r, stale) ------|   if the update arrives after round r closed
  |<--------- SHUTDOWN ----------|
```

* **Aggregation:** `w = sum_k (n_k / sum n) * w_k` over the updates accepted in this round (sample-weighted FedAvg).
* **Round close:** a round closes when every selected worker has replied, a disconnected worker has been
  dropped, or the deadline passes. With `--early-quorum`, it closes as soon as `min_clients` updates are in.
* **Minimum participation:** if fewer than `--min-clients` updates arrived, the round is marked
  `insufficient` and the previous global model is kept. The round id still advances.
* **Stale updates:** an update tagged with an older round id is never aggregated. The coordinator
  logs it, counts it, and sends `REJECT`. A worker that fell behind skips superseded `TRAIN`
  messages and trains on the newest model.
* **Failure detection:** a crash closes the TCP connection, so the coordinator drops that worker
  immediately. A hang or an omitted reply is only caught by the round timeout. This contrast is
  deliberate: crash-stop failures can be detected, omission failures need timeouts.
* **Privacy:** each worker computes its own shard from the shared partition seed. The coordinator
  only loads the test set. Only weights cross the network.
* **Byte accounting:** every frame (header included) is counted on the coordinator's sockets and
  reported per round as `bytes_down` / `bytes_up` / `bytes_total`.

## Data splits

| split | what each worker sees |
|---|---|
| `iid` | uniform random 12k samples |
| `shards` | data sorted by label, cut into 2·N shards, 2 shards each, so ~2 classes per worker |
| `dirichlet` | per-class proportions drawn from Dir(α) (α=0.3 is strongly skewed, with unequal shard sizes) |
| `mixed` | workers 0–1 see only 2 classes each; the rest get an IID mix of the remainder |

## Experiments (`run_all.sh`)

| name | purpose |
|---|---|
| `centralized` | baseline: 1 epoch on all 60k ≙ 1 FL round with E=1 and full participation |
| `iid`, `shards`, `dirichlet`, `mixed` | accuracy gap caused by data heterogeneity |
| `iid_fp16` | half-precision weights on the wire (bytes/round halved) |
| `iid_frac60` | 3 of 5 workers sampled per round |
| `iid_e5`, `shards_e5` | more local computation per round: fewer rounds needed vs client drift |
| `straggler_wait` | worker 4 is always later than the 10 s timeout; each round waits out the deadline |
| `straggler_quorum` | same straggler, but the round closes as soon as 4 updates are in |
| `crash` | worker 3 crashes permanently at round 10; worker 4 crashes and re-joins 20 s later |
| `hang_drop` | worker 2 hangs (connection stays open) from round 10; worker 1 drops 30% of rounds |
| `strict_quorum` | min participation = 5, then one crash: every later round is `insufficient` |
| `crash_shards` | non-IID + crash: the classes held only by worker 0 disappear from training |

## Compute requirements

No cluster or GPU is needed. Everything runs as local processes on one laptop.

| resource | need | measured on the dev machine (Apple M4, 10 cores, 16 GB) |
|---|---|---|
| CPU | 6 processes × 1 thread; 4+ cores is comfortable | 1 local epoch on 12k samples ≈ 1.4 s |
| RAM | ~250–350 MB per process, ~2 GB total | |
| disk | ~60 MB for the dataset + its cache; results are a few MB | |
| network | loopback only; ~1.7 MB per round for 5 workers at fp32 | model = 187 KB (fp32) / 94 KB (fp16) |
| time | ~1–2 min per normal 30-round run; ~5 min for straggler runs; ~25 min for the suite | |
| GPU | not used; tiny batches make per-process CPU faster than shared GPU/MPS here | |
