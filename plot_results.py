"""Turn results/*/metrics.jsonl into report figures and a summary table.

Usage: python plot_results.py [--results results] [--targets 0.98 0.99]
Writes figures to results/figures/ and the table to results/summary.md / summary.csv.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2,
})


def load(results: Path) -> dict:
    runs = {}
    for d in sorted(results.iterdir()):
        m = d / "metrics.jsonl"
        if d.is_dir() and m.exists() and m.stat().st_size and not d.name.startswith("smoke"):
            df = pd.DataFrame([json.loads(line) for line in m.read_text().splitlines()])
            cfg = json.loads((d / "config.json").read_text())
            runs[d.name] = (df, cfg)
    return runs


def first_reaching(df, col, target):
    hit = df[df.test_acc >= target]
    return None if hit.empty else float(hit[col].iloc[0])


def summarize(runs: dict, targets: list) -> pd.DataFrame:
    rows = []
    for name, (df, cfg) in runs.items():
        fed = cfg.get("mode") == "federated"
        row = {
            "experiment": name,
            "split": cfg.get("split", "-") if fed else "centralized",
            "final_acc": df.test_acc.iloc[-1],
            "best_acc": df.test_acc.max(),
        }
        for t in targets:
            row[f"rounds_to_{t:.0%}"] = first_reaching(df, "round", t)
            row[f"time_to_{t:.0%}_s"] = first_reaching(df, "wall_time", t)
        rows.append(row | {
            "mean_round_s": df["round_time"].mean() if fed else df.wall_time.diff().fillna(df.wall_time).mean(),
            "mean_KB_per_round": df.bytes_total.mean() / 1024 if fed else 0.0,
            "total_MB": df.bytes_total.sum() / 2**20 if fed else 0.0,
            "stale_rejected": int(df.stale_rejected.sum()) if fed else 0,
            # selected but not aggregated, excluding detected disconnects (timeouts / early-closed rounds)
            "missing_updates": int((df.selected.map(len) - df.participants.map(len)
                                    - df.dropped.map(len)).sum()) if fed else 0,
            "detected_disconnects": int(df.dropped.map(len).sum()) if fed else 0,
            "insufficient_rounds": int((df.status == "insufficient").sum()) if fed else 0,
        })
    return pd.DataFrame(rows)


def line_chart(ax, runs, names, x="round", labels=None, baseline=True):
    if baseline and "centralized" in runs:
        c = runs["centralized"][0]
        ax.plot(c[x] if x in c else c["round"], c.test_acc, color=BASELINE, ls="--", lw=1.5, label="centralized")
    for i, n in enumerate(n for n in names if n in runs):
        df = runs[n][0]
        ax.plot(df[x], df.test_acc, color=SERIES[i], label=(labels or {}).get(n, n))
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower right")


def save(fig, out: Path, name: str, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", out / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.98, 0.99])
    args = ap.parse_args()
    results = Path(args.results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load(results)

    table = summarize(runs, args.targets)
    table.to_csv(results / "summary.csv", index=False)
    (results / "summary.md").write_text(table.to_markdown(index=False, floatfmt=".4g"))
    print(table.to_string(index=False))

    # 1. Data heterogeneity: IID vs non-IID vs centralized.
    fig, ax = plt.subplots(figsize=(7, 4))
    line_chart(ax, runs, ["iid", "dirichlet", "mixed", "shards"],
               labels={"iid": "IID", "dirichlet": "Dirichlet α=0.3", "mixed": "mixed (2 workers see 2 classes)",
                       "shards": "shards (each worker ~2 classes)"})
    ax.set(xlabel="round (= epochs of data for centralized)", title="Accuracy vs round: data heterogeneity")
    ax.set_ylim(0.8, 1.0)
    save(fig, out, "accuracy_splits.png")

    # 2. Local epochs: more local work per round under IID vs non-IID.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (a, b), title in zip(axes, [("iid", "iid_e5"), ("shards", "shards_e5")], ["IID", "non-IID shards"]):
        line_chart(ax, runs, [a, b], labels={a: "E=1 local epoch", b: "E=5 local epochs"}, baseline=False)
        ax.set(xlabel="round", title=f"{title}: local epochs per round")
    axes[0].set_ylim(0.8, 1.0)
    save(fig, out, "local_epochs.png")

    # 3. Communication cost.
    comm = [n for n in ["iid", "iid_fp16", "iid_frac60", "iid_e5", "straggler_wait", "crash"] if n in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    kb = [runs[n][0].bytes_total.mean() / 1024 for n in comm]
    axes[0].barh(comm[::-1], kb[::-1], color=SERIES[0], height=0.6)
    for y, v in enumerate(kb[::-1]):
        axes[0].text(v, y, f" {v:,.0f} KB", va="center", color=INK2, fontsize=9)
    axes[0].set(xlabel="mean bytes per round (KB, up + down)", title="Communication per round")
    axes[0].grid(axis="y", visible=False)
    for i, n in enumerate(n for n in ["iid", "iid_fp16", "iid_frac60", "iid_e5"] if n in runs):
        df = runs[n][0]
        axes[1].plot(df.bytes_total.cumsum() / 2**20, df.test_acc, color=SERIES[i], label=n)
    axes[1].set(xlabel="cumulative traffic (MB)", ylabel="test accuracy", title="Accuracy vs bytes transferred",
                ylim=(0.95, 1.0))
    axes[1].legend(loc="lower right")
    save(fig, out, "communication.png")

    # 4. Faults: accuracy per round and over wall-clock time.
    faults = ["iid", "straggler_wait", "straggler_quorum", "crash", "hang_drop", "strict_quorum"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    line_chart(axes[0], runs, faults, baseline=False)
    axes[0].set(xlabel="round", title="Accuracy vs round under faults", ylim=(0.9, 1.0))
    line_chart(axes[1], runs, faults, x="wall_time", baseline=False)
    axes[1].set(xlabel="wall-clock time (s)", title="Accuracy vs time under faults")
    save(fig, out, "faults_accuracy.png")

    # 5. Participation per round for fault scenarios (small multiples, one axis each).
    fnames = [n for n in faults[1:] + ["crash_shards"] if n in runs]
    if fnames:
        fig, axes = plt.subplots(len(fnames), 1, figsize=(8, 1.5 * len(fnames)), sharex=True, squeeze=False)
        for ax, n in zip(axes[:, 0], fnames):
            df = runs[n][0]
            ok = df.status == "aggregated"
            ax.bar(df["round"][ok], df.participants.map(len)[ok], color=SERIES[0], width=0.8, label="aggregated")
            ax.bar(df["round"][~ok], df.participants.map(len)[~ok], color=SERIES[7], width=0.8,
                   label="below min - skipped")
            ax.axhline(runs[n][1]["min_clients"], color=INK2, lw=1, ls=":")
            ax.set_ylabel(n, rotation=0, ha="right", va="center", color=INK)
            ax.set_ylim(0, runs[n][1]["num_workers"] + 0.5)
            ax.grid(axis="x", visible=False)
        handles = [matplotlib.patches.Patch(color=SERIES[0], label="aggregated"),
                   matplotlib.patches.Patch(color=SERIES[7], label="below min participation - skipped"),
                   matplotlib.lines.Line2D([], [], color=INK2, lw=1, ls=":", label="min participation")]
        fig.suptitle("Updates accepted per round", color=INK, fontsize=11, y=0.995)
        fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.97))
        axes[-1, 0].set_xlabel("round")
        save(fig, out, "faults_participation.png", rect=(0, 0, 1, 0.94))

    # 6. Non-IID + failure: losing the only holder of some classes.
    fig, ax = plt.subplots(figsize=(7, 4))
    line_chart(ax, runs, ["shards", "crash_shards"],
               labels={"shards": "shards, no faults", "crash_shards": "shards, worker 0 crashes at round 10"})
    ax.set(xlabel="round", title="Non-IID: a crashed worker takes its classes with it", ylim=(0.0, 1.0))
    save(fig, out, "crash_noniid.png")


if __name__ == "__main__":
    main()
