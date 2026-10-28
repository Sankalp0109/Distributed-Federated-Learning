"""Centralized baseline: same model and optimizer, trained on the full training set.

One epoch here processes the same number of samples as one FL round with
local_epochs=1 and full participation, so epochs and rounds line up on the x-axis.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .data import load_raw, normalize
from .model import SmallCNN, evaluate, train_epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/centralized")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"mode": "centralized"}, indent=2))

    tx, ty, ex, ey = load_raw(args.dataset)
    x, test_x = normalize(tx, args.dataset), normalize(ex, args.dataset)
    model = SmallCNN()
    gen = torch.Generator().manual_seed(args.seed)
    start = time.time()
    with open(out / "metrics.jsonl", "w") as f:
        for ep in range(1, args.epochs + 1):
            loss = train_epochs(model, x, ty, 1, args.lr, args.batch_size, generator=gen)
            acc, tloss = evaluate(model, test_x, ey)
            rec = {"round": ep, "test_acc": acc, "test_loss": tloss, "mean_train_loss": loss,
                   "wall_time": time.time() - start, "bytes_total": 0, "status": "aggregated"}
            f.write(json.dumps(rec) + "\n")
            print(f"epoch {ep:3d} acc={acc:.4f} loss={tloss:.4f} ({rec['wall_time']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
