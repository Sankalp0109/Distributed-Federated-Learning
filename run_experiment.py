"""Launch one FL experiment: a coordinator process plus N worker processes.

Example:
  python run_experiment.py --name noniid_shards --split shards --rounds 30
  python run_experiment.py --name straggler --slow 4:12 --min-clients 3
  python run_experiment.py --name crash --crash 3:8 --crash-recover 4:5:20

Fault flags take WORKER:ARGS and can be repeated:
  --slow W:DELAY[:JITTER]     add DELAY(+U[0,JITTER]) seconds before each reply
  --crash W:ROUND             worker process exits when it receives ROUND
  --crash-recover W:ROUND:S   worker drops out at ROUND, re-joins after S seconds
  --hang W:ROUND              worker stays connected but never replies from ROUND on
  --drop W:P                  worker silently skips each round with probability P
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from fl.data import SCHEMES, label_histogram, load_raw, partition_from_config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_faults(args) -> dict:
    faults: dict = {}

    def add(w, **kv):
        faults.setdefault(str(int(w)), {}).update(kv)

    for s in args.slow:
        w, *rest = s.split(":")
        add(w, delay=float(rest[0]), jitter=float(rest[1]) if len(rest) > 1 else 0.0)
    for s in args.crash:
        w, r = s.split(":")
        add(w, crash_at_round=int(r))
    for s in args.crash_recover:
        w, r, t = s.split(":")
        add(w, crash_at_round=int(r), recover_after=float(t))
    for s in args.hang:
        w, r = s.split(":")
        add(w, hang_at_round=int(r))
    for s in args.drop:
        w, p = s.split(":")
        add(w, drop_prob=float(p))
    return faults


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--split", default="iid", choices=SCHEMES)
    ap.add_argument("--classes-per-worker", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration")
    ap.add_argument("--n-skewed", type=int, default=2, help="workers restricted to few classes in 'mixed'")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--round-timeout", type=float, default=10.0)
    ap.add_argument("--min-clients", type=int, default=None, help="default: ceil(workers/2)")
    ap.add_argument("--early-quorum", action="store_true",
                    help="close a round as soon as min-clients updates arrive")
    ap.add_argument("--clients-fraction", type=float, default=1.0)
    ap.add_argument("--wire-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow", action="append", default=[])
    ap.add_argument("--crash", action="append", default=[])
    ap.add_argument("--crash-recover", action="append", default=[])
    ap.add_argument("--hang", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    out = Path(args.results) / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": args.name, "mode": "federated", "dataset": args.dataset, "num_workers": args.workers,
        "split": args.split, "classes_per_worker": args.classes_per_worker, "alpha": args.alpha,
        "n_skewed": args.n_skewed, "rounds": args.rounds, "local_epochs": args.local_epochs,
        "lr": args.lr, "batch_size": args.batch_size, "round_timeout": args.round_timeout,
        "min_clients": args.min_clients or -(-args.workers // 2), "early_quorum": args.early_quorum,
        "clients_fraction": args.clients_fraction, "wire_dtype": args.wire_dtype,
        "threads": args.threads, "seed": args.seed, "host": "127.0.0.1", "port": free_port(),
        "register_timeout": 60.0, "faults": parse_faults(args),
    }
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Record the shard sizes / label histograms for the report (the launcher only, never the coordinator).
    _, train_y, _, _ = load_raw(args.dataset)
    parts = partition_from_config(train_y.numpy(), cfg)
    (out / "partition.json").write_text(json.dumps(
        {"sizes": [len(p) for p in parts], "label_hist": label_histogram(train_y.numpy(), parts)}))

    py = sys.executable
    coord_log = open(out / "coordinator.log", "w")
    coord = subprocess.Popen([py, "-m", "fl.coordinator", "--config", str(cfg_path)],
                             stdout=coord_log, stderr=subprocess.STDOUT)
    time.sleep(1.0)
    workers = []
    for w in range(args.workers):
        wlog = open(out / f"worker_{w}.log", "w")
        workers.append(subprocess.Popen([py, "-m", "fl.worker", "--config", str(cfg_path), "--worker-id", str(w)],
                                        stdout=wlog, stderr=subprocess.STDOUT))
    print(f"[{args.name}] coordinator pid {coord.pid}, workers {[p.pid for p in workers]}; logs in {out}/",
          flush=True)
    try:
        rc = coord.wait()
    finally:
        time.sleep(0.5)
        for p in workers:  # hung or still-recovering workers do not exit on their own
            if p.poll() is None:
                p.terminate()
        for p in workers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    lines = (out / "metrics.jsonl").read_text().splitlines() if (out / "metrics.jsonl").exists() else []
    if lines:
        last = json.loads(lines[-1])
        print(f"[{args.name}] exit={rc} rounds={len(lines)} final_acc={last['test_acc']:.4f} "
              f"wall={last['wall_time']:.1f}s", flush=True)
    else:
        print(f"[{args.name}] exit={rc}, no metrics written - see {out}/coordinator.log", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
