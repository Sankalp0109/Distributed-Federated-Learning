"""Dataset loading and IID / non-IID partitioning.

Partitioning is a pure function of (labels, scheme, seed), so each worker can
compute its own shard locally: the coordinator never touches training data.
"""
from pathlib import Path

import numpy as np
import torch

_STATS = {"mnist": (0.1307, 0.3081), "fashion_mnist": (0.2860, 0.3530)}
SCHEMES = ("iid", "shards", "dirichlet", "mixed")


def load_raw(name: str, root: str = "data"):
    """Returns uint8 images and int64 labels: (train_x, train_y, test_x, test_y)."""
    from torchvision import datasets

    cache = Path(root) / f"{name}.pt"
    if not cache.exists():
        cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
        tr = cls(root, train=True, download=True)
        te = cls(root, train=False, download=True)
        Path(root).mkdir(parents=True, exist_ok=True)
        torch.save({"train_x": tr.data, "train_y": tr.targets,
                    "test_x": te.data, "test_y": te.targets}, cache)
    d = torch.load(cache)
    return d["train_x"], d["train_y"].long(), d["test_x"], d["test_y"].long()


def normalize(x_uint8: torch.Tensor, name: str) -> torch.Tensor:
    mean, std = _STATS[name]
    return ((x_uint8.float() / 255.0 - mean) / std).unsqueeze(1)


def partition(labels, n_workers: int, scheme: str, seed: int = 0,
              classes_per_worker: int = 2, alpha: float = 0.5, n_skewed: int = 2) -> list:
    """Split sample indices across workers. Returns a list of index arrays.

    iid       - uniform random split.
    shards    - McMahan et al.: sort by label, cut into n*k shards, give each
                worker k shards, so each worker sees ~k classes.
    dirichlet - per-class proportions drawn from Dir(alpha); small alpha = more skew.
    mixed     - the first `n_skewed` workers see only `classes_per_worker`
                classes; the remaining workers get an IID mix of what is left.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    num_classes = int(labels.max()) + 1

    if scheme == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), n_workers)]

    if scheme == "shards":
        order = np.argsort(labels, kind="stable")
        shards = np.array_split(order, n_workers * classes_per_worker)
        shard_ids = rng.permutation(len(shards))
        return [np.sort(np.concatenate([shards[s] for s in
                                        shard_ids[w * classes_per_worker:(w + 1) * classes_per_worker]]))
                for w in range(n_workers)]

    if scheme == "dirichlet":
        parts = [[] for _ in range(n_workers)]
        for c in range(num_classes):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet([alpha] * n_workers)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for w, chunk in enumerate(np.split(idx, cuts)):
                parts[w].extend(chunk.tolist())
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]

    if scheme == "mixed":
        pools = {c: list(rng.permutation(np.where(labels == c)[0])) for c in range(num_classes)}
        per_worker = n // n_workers
        parts = []
        for w in range(n_skewed):
            classes = [(w * classes_per_worker + j) % num_classes for j in range(classes_per_worker)]
            take = per_worker // classes_per_worker
            chosen = []
            for c in classes:
                chosen.extend(pools[c][:take])
                pools[c] = pools[c][take:]
            parts.append(np.sort(np.array(chosen, dtype=np.int64)))
        rest = rng.permutation(np.concatenate([np.array(v, dtype=np.int64) for v in pools.values()]))
        parts.extend(np.sort(p) for p in np.array_split(rest, n_workers - n_skewed))
        return parts

    raise ValueError(f"unknown partition scheme {scheme!r}; choose from {SCHEMES}")


def partition_from_config(labels, cfg: dict) -> list:
    return partition(labels, cfg["num_workers"], cfg["split"], seed=cfg["seed"],
                     classes_per_worker=cfg["classes_per_worker"], alpha=cfg["alpha"],
                     n_skewed=cfg["n_skewed"])


def label_histogram(labels, parts) -> list:
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    return [np.bincount(labels[p], minlength=num_classes).tolist() for p in parts]
