"""Federated-learning worker.

Holds a private data shard (computed locally from the shared partition seed),
trains the received global model for a few local epochs, and sends back the
weights tagged with the round id. Supports fault injection for experiments:
slow (fixed delay + jitter), crash (optionally recovering), hang, and drop.
"""
import argparse
import json
import logging
import os
import queue
import random
import threading
import time
from pathlib import Path

import torch

from . import protocol as P
from .data import load_raw, normalize, partition_from_config
from .model import SmallCNN, get_weights, set_weights, train_epochs

log = logging.getLogger("worker")


class Worker:
    def __init__(self, cfg: dict, wid: int):
        self.cfg = cfg
        self.wid = wid
        self.faults = cfg.get("faults", {}).get(str(wid), {})
        self.rng = random.Random(cfg["seed"] * 1000 + wid)
        self.gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + wid)

        train_x, train_y, _, _ = load_raw(cfg["dataset"])
        idx = torch.from_numpy(partition_from_config(train_y.numpy(), cfg)[wid])
        self.x = normalize(train_x[idx], cfg["dataset"])
        self.y = train_y[idx]
        self.model = SmallCNN()
        self.crash_pending = "crash_at_round" in self.faults
        log.info("worker %d: %d local samples, faults=%s", wid, len(self.x), self.faults)

    def connect(self) -> P.Channel:
        for _ in range(100):
            try:
                ch = P.Channel.connect(self.cfg["host"], self.cfg["port"])
                ch.send({"type": P.REGISTER, "worker_id": self.wid, "num_samples": len(self.x)})
                assert ch.recv()["type"] == P.REGISTER_ACK
                return ch
            except (ConnectionError, OSError):
                time.sleep(0.2)
        raise SystemExit(f"worker {self.wid}: could not reach coordinator")

    @staticmethod
    def _reader(ch: P.Channel, q: queue.Queue):
        try:
            while True:
                q.put(ch.recv())
        except (ConnectionError, OSError, EOFError):
            q.put({"type": P.DISCONNECT})

    def session(self) -> str:
        """One connection lifetime. Returns 'shutdown', 'lost' or 'crashed'."""
        ch = self.connect()
        log.info("connected and registered")
        q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, args=(ch, q), daemon=True).start()
        try:
            while True:
                msgs = [q.get()]
                while not q.empty():
                    msgs.append(q.get_nowait())
                types = {m["type"] for m in msgs}
                if P.SHUTDOWN in types:
                    log.info("shutdown received")
                    return "shutdown"
                if P.DISCONNECT in types:
                    log.warning("connection lost")
                    return "lost"
                for m in msgs:
                    if m["type"] == P.REJECT:
                        log.info("coordinator rejected my update for round %d (%s)", m["round_id"], m["reason"])
                trains = [m for m in msgs if m["type"] == P.TRAIN]
                if not trains:
                    continue
                if len(trains) > 1:
                    # We fell behind: skip superseded rounds and work on the newest model.
                    log.info("skipping superseded rounds %s", [m["round_id"] for m in trains[:-1]])
                outcome = self.do_round(ch, trains[-1])
                if outcome:
                    return outcome
        finally:
            ch.close()

    def do_round(self, ch: P.Channel, msg: dict):
        r = msg["round_id"]
        f = self.faults
        if self.crash_pending and r >= f["crash_at_round"]:
            self.crash_pending = False
            log.warning("injected CRASH at round %d", r)
            return "crashed"
        if f.get("hang_at_round") is not None and r >= f["hang_at_round"]:
            log.warning("injected HANG at round %d (connection stays open, no replies)", r)
            while True:
                time.sleep(3600)
        if f.get("drop_prob", 0) and self.rng.random() < f["drop_prob"]:
            log.info("round %d: injected DROP (no reply)", r)
            return None

        set_weights(self.model, msg["weights"])
        t0 = time.time()
        loss = train_epochs(self.model, self.x, self.y, msg["epochs"], msg["lr"], msg["batch_size"],
                            generator=self.gen)
        train_time = time.time() - t0
        delay = f.get("delay", 0.0) + self.rng.uniform(0, f.get("jitter", 0.0))
        if delay:
            time.sleep(delay)
        ch.send({"type": P.UPDATE, "round_id": r, "worker_id": self.wid,
                 "weights": get_weights(self.model, self.cfg["wire_dtype"]),
                 "num_samples": len(self.x), "train_loss": loss, "train_time": train_time})
        log.info("round %d: sent update (loss %.4f, train %.2fs, delay %.2fs)", r, loss, train_time, delay)
        return None

    def run(self):
        while True:
            outcome = self.session()
            if outcome == "shutdown":
                return
            if outcome == "crashed":
                recover = self.faults.get("recover_after")
                if recover is None:
                    log.warning("crashed permanently")
                    os._exit(1)
                log.warning("down for %.1fs, then re-joining", recover)
                time.sleep(recover)
                continue
            if outcome == "lost":
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--worker-id", type=int, required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s worker{args.worker_id} %(levelname)s %(message)s")
    torch.set_num_threads(cfg["threads"])
    Worker(cfg, args.worker_id).run()


if __name__ == "__main__":
    main()
