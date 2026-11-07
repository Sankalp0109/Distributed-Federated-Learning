# Model & Training-Time Comparison

## Model: `SmallCNN` (`fl/model.py`)

A small CNN for 28x28 grayscale images (MNIST / Fashion-MNIST), shared by **every**
training mode below (centralized and federated) so accuracy/time comparisons are
apples-to-apples.

| layer | shape | params |
|---|---|---:|
| `conv1` | Conv2d(1 → 16, kernel 5x5) | 416 |
| `conv2` | Conv2d(16 → 32, kernel 5x5) | 12,832 |
| `fc1` | Linear(512 → 64) | 32,832 |
| `fc2` | Linear(64 → 10) | 650 |
| **total** | | **46,730** |

Forward pass: `conv1 -> ReLU -> maxpool(2) -> conv2 -> ReLU -> maxpool(2) -> flatten -> fc1 -> ReLU -> fc2`
(28→24→12→8→4, flattened to 32·4·4 = 512).

- **Optimizer:** plain SGD, `lr=0.01`, `momentum=0.9`
- **Batch size:** 32
- **Loss:** cross-entropy
- **Wire size:** 187 KB per model copy at fp32, 94 KB at fp16 (`--wire-dtype`)

## Two ways this model gets trained

| | Centralized (sequential) | Federated (distributed) |
|---|---|---|
| process | 1 process, 1 dataset | 1 coordinator + 4–6 worker processes |
| data per step | full 60k training set, shuffled each epoch | each worker's own private shard (e.g. 12k for 5 IID workers) |
| 1 "step" | 1 epoch over all 60k samples | 1 round = broadcast → local SGD (`local_epochs`) on each worker **in parallel** → sample-weighted FedAvg |
| network | none | TCP, framed messages, byte-counted |
| script | `fl/centralized.py` | `fl/coordinator.py` + `fl/worker.py` via `run_experiment.py` |

Because the federated workers train **in parallel** on smaller shards, one federated
round is cheaper in wall-clock time than one centralized epoch over the full dataset,
even though the workers collectively process a comparable number of samples.

## Measured time difference (from `results/summary.md`)

Same model, same machine, same optimizer settings. "round/epoch" = coordinator round
for federated runs, full-dataset epoch for centralized.

| run | mode / split | mean time per round-or-epoch (s) | rounds to 98% acc | time to 98% (s) | rounds to 99% acc | time to 99% (s) | final acc |
|---|---|---:|---:|---:|---:|---:|---:|
| `centralized` | sequential, full 60k | 6.246 | 1 | 6.03 | 3 | 18.73 | 0.9934 |
| `iid` | federated, 5 workers, IID | 2.684 | 2 | 5.18 | 6 | 15.89 | 0.9921 |
| `iid_e5` | federated, IID, 5 local epochs/round | 14.37 | 1 | 13.04 | 2 | 28.37 | 0.9925 |
| `iid_frac60` | federated, 3/5 workers sampled per round | 2.295 | 2 | 4.37 | 8 | 18.27 | 0.9929 |
| `shards` | federated, ~2 classes/worker (non-IID) | 2.641 | 22 | 57.98 | — | — | 0.9811 |
| `dirichlet` | federated, Dirichlet(α=0.3) skew | 3.634 | 4 | 15.13 | 9 | 33.26 | 0.9913 |
| `mixed` | federated, 2 workers 2-class / rest IID | 2.780 | 14 | 38.07 | — | — | 0.9872 |

**Takeaways:**
- **Per-step cost:** a federated round (`~2.6–2.7 s` for IID) is roughly **2.3x faster**
  than one centralized epoch (`6.25 s`) because the 60k samples are split across 5
  worker processes training concurrently instead of one process working through all
  of them sequentially.
- **Steps to converge:** centralized still reaches 98%/99% accuracy in fewer, cheaper
  steps when data is IID, because every centralized epoch already sees the full
  dataset, while a federated round only aggregates partial, locally-trained updates
  (client drift). Wall-clock-to-accuracy ends up close for IID splits (`~5-6 s` to 98%
  either way) but federated still wins at the 99% mark on raw per-round speed.
- **Non-IID hurts the federated side specifically:** `shards` and `mixed` need 7–14x
  more rounds than `iid` to hit 98% accuracy, since each worker's local update is
  biased toward the few classes it holds (client drift grows with heterogeneity) —
  this is a cost that sequential/centralized training never pays, since it never
  partitions data at all.
- **More local computation per round (`iid_e5`, 5 local epochs instead of 1) cuts the
  number of rounds needed roughly in half but makes each round ~5x longer**, so total
  time-to-accuracy is similar to doing more, cheaper rounds (`iid`).

## Suggested models to demonstrate the advantage of this training method

The `SmallCNN` runs above only take ~2.6–6 s per step, so the sequential-vs-distributed
time gap, while real (~2.3x), is not dramatic — it's hard to convince a professor with
a few-second difference. The models below are picked to make that gap undeniable: each
one is heavy enough per training step that **running it sequentially on the full
dataset is noticeably slow**, so splitting the same work across parallel workers (this
project's coordinator/worker setup) produces a large, visible wall-clock speedup —
while still finishing in minutes, not hours, on an Apple laptop (M-series CPU, and
optionally the `mps` backend in PyTorch for the conv/transformer ones).

| # | model | params (≈) | dataset | why it shows the difference |
|---|---|---:|---|---|
| 1 | **ResNet-9** (or `torchvision` ResNet-18) | 2.5M (ResNet-9) / 11M (ResNet-18) | CIFAR-10 | a real, commonly-cited FL benchmark (used in FedAvg/FedProx papers); a centralized epoch over 50k 32x32x3 images is slow enough on CPU/MPS that distributing it across 5 workers produces a clear, citable wall-clock speedup, and CIFAR-10 is hard enough that the non-IID accuracy gap (vs. the easy MNIST case) becomes obvious too |
| 2 | **MobileNetV2** (width multiplier 0.5, scaled down) | ~1–2M | CIFAR-10 or a small Tiny-ImageNet subset | MobileNet is literally designed for the on-device/edge scenario federated learning targets, so it doubles as a "this is the actual use case" argument, not just a speed demo; still light enough to fine-tune quickly on a laptop |
| 3 | **DistilBERT** (or a 2–4 layer Transformer trained from scratch, if fine-tuning a pretrained model feels like cheating) | ~66M (DistilBERT) / ~1–5M (small Transformer) | text classification (AG News or IMDB sentiment) | the current model's weights are 187 KB — too small to make the *communication cost* story compelling. A model this size makes bytes-per-round (and the fp16-halving trick already in `fl/model.py`) a real, visible bottleneck, which is the other half of what this project measures besides time |
| 4 | **2-layer LSTM/GRU character-level language model** | ~1–5M (depends on hidden size/vocab) | Shakespeare text, split **by character/play** (a classic FedAvg-paper benchmark) | sequential RNN training is inherently slower per batch than a CNN forward/backward, so centralized training time grows a lot while federated per-worker shards stay small — good second axis (sequence data, not images) to show the method isn't CNN-specific |

**Recommendation:** lead with **#1 (ResNet-9/18 on CIFAR-10)** — it is the closest
drop-in to the current MNIST pipeline (still an image classifier, still works with
`fl/data.py`-style loading), it is the model size/dataset combination most reviewers
recognize as a standard FL benchmark, and the per-epoch cost on CPU/MPS is large enough
to make the sequential-vs-distributed timing chart (like the `mean_round_s` table
above) much more persuasive. Use **#3 (DistilBERT / small Transformer)** as the second
example specifically to show the communication-cost advantage, since that's where the
current tiny CNN under-sells the method.
