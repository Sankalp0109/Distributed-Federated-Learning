"""Federated-learning coordinator.

Accepts worker connections, runs synchronous FedAvg rounds with round ids and
deadlines, rejects late (stale) updates, and only aggregates when a stated
minimum number of workers reported in time. Writes one JSON line per round.
"""
import argparse
import json
import logging
import math
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from . import protocol as P
from .data import load_raw, normalize
from .model import SmallCNN, evaluate, fedavg, get_weights, num_params, set_weights, to_wire

log = logging.getLogger("coordinator")


@dataclass
class WorkerConn:
    channel: P.Channel
    num_samples: int
    alive: bool = True


class Coordinator:
    def __init__(self, cfg: dict, out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.workers: dict[int, WorkerConn] = {}
        self.channels: list[P.Channel] = []  # every channel ever opened, for byte totals
        self.lock = threading.Lock()
        self.inbox: queue.Queue = queue.Queue()
        self.stopping = False
        self.rng = random.Random(cfg["seed"])

        torch.manual_seed(cfg["seed"])
        self.model = SmallCNN()
        self.global_weights = get_weights(self.model)
        _, _, test_x, test_y = load_raw(cfg["dataset"])
        self.test_x, self.test_y = normalize(test_x, cfg["dataset"]), test_y

    # ---------------------------------------------------------------- networking
    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.cfg["host"], self.cfg["port"]))
        self.server.listen()
        threading.Thread(target=self._accept_loop, daemon=True).start()
        log.info("listening on %s:%d", self.cfg["host"], self.cfg["port"])

    def _accept_loop(self):
        while not self.stopping:
            try:
                sock, _ = self.server.accept()
            except OSError:
                return
            ch = P.Channel(sock)
            with self.lock:
                self.channels.append(ch)
            threading.Thread(target=self._reader, args=(ch,), daemon=True).start()

    def _reader(self, ch: P.Channel):
        """One thread per connection: handles REGISTER, forwards everything else to the round loop."""
        wid = None
        try:
            while True:
                msg = ch.recv()
                if msg["type"] == P.REGISTER:
                    wid = msg["worker_id"]
                    with self.lock:
                        old = self.workers.get(wid)
                        rejoin = old is not None
                        self.workers[wid] = WorkerConn(ch, msg["num_samples"])
                    if old is not None and old.channel is not ch:
                        old.channel.close()
                    ch.send({"type": P.REGISTER_ACK})
                    log.info("worker %d %s (%d samples)", wid, "re-joined" if rejoin else "registered",
                             msg["num_samples"])
                else:
                    self.inbox.put((wid, msg))
        except (ConnectionError, OSError, EOFError):
            pass
        if wid is not None:
            with self.lock:
                w = self.workers.get(wid)
                if w is not None and w.channel is ch:
                    w.alive = False
                    log.warning("worker %d disconnected", wid)
            self.inbox.put((wid, {"type": P.DISCONNECT, "channel": ch}))

    def total_bytes(self):
        with self.lock:
            return (sum(c.bytes_sent for c in self.channels), sum(c.bytes_recv for c in self.channels))

    def alive_workers(self):
        with self.lock:
            return sorted(wid for wid, w in self.workers.items() if w.alive)

    def wait_for_workers(self):
        deadline = time.time() + self.cfg["register_timeout"]
        while time.time() < deadline and len(self.alive_workers()) < self.cfg["num_workers"]:
            time.sleep(0.1)
        log.info("starting with %d/%d workers: %s", len(self.alive_workers()),
                 self.cfg["num_workers"], self.alive_workers())

    # ---------------------------------------------------------------- rounds
    def run_round(self, r: int) -> dict:
        cfg = self.cfg
        t0 = time.time()
        sent0, recv0 = self.total_bytes()

        alive = self.alive_workers()
        k = max(1, math.ceil(cfg["clients_fraction"] * len(alive))) if alive else 0
        selected = sorted(self.rng.sample(alive, k)) if k else []

        msg = {"type": P.TRAIN, "round_id": r,
               "weights": to_wire(self.global_weights, cfg["wire_dtype"]),
               "epochs": cfg["local_epochs"], "lr": cfg["lr"], "batch_size": cfg["batch_size"],
               "deadline_s": cfg["round_timeout"]}
        pending = set()
        for wid in selected:
            with self.lock:
                ch = self.workers[wid].channel
            try:
                ch.send(msg)
                pending.add(wid)
            except OSError:
                log.warning("round %d: send to worker %d failed", r, wid)

        updates, stale, stale_from, dropped = {}, 0, [], []
        deadline = t0 + cfg["round_timeout"]
        quorum = cfg["min_clients"]

        def handle(wid, m) -> None:
            nonlocal stale
            if m["type"] == P.DISCONNECT:
                with self.lock:
                    current = self.workers[wid].channel is m["channel"]
                if current and wid in pending:
                    pending.discard(wid)
                    dropped.append(wid)
                return
            if m["type"] != P.UPDATE:
                return
            if m["round_id"] != r:
                # Late update from an earlier round: reject it, never aggregate it.
                stale += 1
                stale_from.append(wid)
                log.info("round %d: rejected stale update from worker %d (round %d)", r, wid, m["round_id"])
                with self.lock:
                    ch = self.workers[wid].channel if wid in self.workers else None
                try:
                    if ch is not None:
                        ch.send({"type": P.REJECT, "round_id": m["round_id"], "reason": "stale"})
                except OSError:
                    pass
                return
            if wid in pending:
                updates[wid] = m
                pending.discard(wid)

        while pending:
            if cfg["early_quorum"] and len(updates) >= quorum:
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                handle(*self.inbox.get(timeout=remaining))
            except queue.Empty:
                break
        # Count anything that already arrived (e.g. stale updates) without waiting further.
        while True:
            try:
                wid, m = self.inbox.get_nowait()
            except queue.Empty:
                break
            if m["type"] == P.UPDATE and m["round_id"] == r and wid not in updates:
                m = dict(m, round_id=-1)  # arrived after we closed the round: treat as late
            handle(wid, m)
        # Workers still pending either missed the deadline or were not waited for (early quorum).
        deadline_hit = time.time() >= deadline
        timed_out = sorted(pending) if deadline_hit else []
        not_waited = [] if deadline_hit else sorted(pending)

        if len(updates) >= quorum:
            self.global_weights = fedavg([(u["weights"], u["num_samples"]) for u in updates.values()])
            status = "aggregated"
        else:
            status = "insufficient"
            log.warning("round %d: only %d/%d updates (min %d) - keeping previous global model",
                        r, len(updates), len(selected), quorum)
        set_weights(self.model, self.global_weights)
        acc, loss = evaluate(self.model, self.test_x, self.test_y)
        sent1, recv1 = self.total_bytes()

        rec = {
            "round": r, "status": status, "test_acc": acc, "test_loss": loss,
            "selected": selected, "participants": sorted(updates), "timed_out": timed_out,
            "not_waited": not_waited,
            "dropped": sorted(dropped), "stale_rejected": stale, "stale_from": stale_from,
            "samples_aggregated": sum(u["num_samples"] for u in updates.values()),
            "mean_train_loss": (sum(u["train_loss"] for u in updates.values()) / len(updates)) if updates else None,
            "max_train_time": max((u["train_time"] for u in updates.values()), default=None),
            "bytes_down": sent1 - sent0, "bytes_up": recv1 - recv0, "bytes_total": (sent1 - sent0) + (recv1 - recv0),
            "round_time": time.time() - t0,
        }
        log.info("round %3d %-12s acc=%.4f loss=%.4f part=%s timeout=%s dropped=%s stale=%d bytes=%d %.2fs",
                 r, status, acc, loss, rec["participants"], timed_out, rec["dropped"], stale,
                 rec["bytes_total"], rec["round_time"])
        return rec

    def run(self):
        self.start_server()
        self.wait_for_workers()
        log.info("model: %d params, wire dtype %s", num_params(self.model), self.cfg["wire_dtype"])
        start = time.time()
        with open(self.out_dir / "metrics.jsonl", "w") as f:
            for r in range(1, self.cfg["rounds"] + 1):
                rec = self.run_round(r)
                rec["wall_time"] = time.time() - start
                f.write(json.dumps(rec) + "\n")
                f.flush()
        self.shutdown()

    def shutdown(self):
        self.stopping = True
        with self.lock:
            conns = [w.channel for w in self.workers.values() if w.alive]
        for ch in conns:
            try:
                ch.send({"type": P.SHUTDOWN})
            except OSError:
                pass
        time.sleep(0.2)
        self.server.close()
        log.info("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(cfg["threads"])
    Coordinator(cfg, cfg_path.parent).run()


if __name__ == "__main__":
    main()
