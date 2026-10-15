"""Poster figures from results/*.json -> figures/*.svg (+ .png previews).

Sized for an A0 landscape poster: each figure is drawn at its printed width
(~14 in for a column) with 20-24 pt text, so nothing is scaled down on print.

Colour follows the method everywhere: ours on GPU = blue, official = orange,
ours on CPU = aqua, no PASTA = grey dashed (reference line, not a method).
Palette: categorical slots 1-3 of the dataviz reference palette (light mode),
which are documented as validated all-pairs for CVD and normal vision.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

C = {"gpu": "#2a78d6", "official": "#eb6834", "ours_cpu": "#1baf7a", "none": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"gpu": "FlashPASTA, on GPU", "cpu_official": "Official, in CPU workers",
         "cpu_ours": "FlashPASTA, in CPU workers", "none": "No PASTA"}
MODE_COLOR = {"gpu": C["gpu"], "cpu_official": C["official"], "cpu_ours": C["ours_cpu"], "none": C["none"]}
MODES = ["none", "cpu_official", "cpu_ours", "gpu"]
COL_W = 14.0  # inches, one poster column
LEGEND_HANDLE = 3.0  # long enough that the dashed no-PASTA line reads as dashed beside the marker

plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 22, "axes.titlesize": 24, "axes.labelsize": 22,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 150, "svg.fonttype": "path", "axes.titlelocation": "left", "axes.titlepad": 14,
    "svg.hashsalt": "flashpasta",  # stable element ids, so regenerated SVGs only differ where the plot does
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.1, metadata={"Date": None})
    plt.close(fig)
    print("wrote", FIG / f"{name}.svg")


def line(ax, xs, ys, color, label, dashed=False):
    ax.plot(xs, ys, color=color, lw=3, ls=(0, (5, 3)) if dashed else "-", marker="o", ms=11,
            mec="white", mew=2, label=label, zorder=3)


def log2_workers(ax, ws):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws, [str(w) for w in ws])
    ax.minorticks_off()
    ax.set_xlabel("CPU data-loader workers")


# ---------------------------------------------------------------- kernel
RES_ORDER = ["640x640", "1280x1280", "sim10k_1052x1914", "4k_2160x3840"]
RES_NAME = {"640x640": "640²", "1280x1280": "1280²", "sim10k_1052x1914": "Sim10k", "4k_2160x3840": "4K"}
RES_MP = {"640x640": 0.41, "1280x1280": 1.64, "sim10k_1052x1914": 2.01, "4k_2160x3840": 8.29}


def kernel_best(rows, res, variant, procs=None):
    r = [x["img_per_s"] for x in rows if x["res"] == res and x["variant"] == variant and x["img_per_s"]
         and (procs is None or x.get("cpu_procs") == procs)]
    return max(r)


def fig_kernel():
    path = RES / "kernel.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    maxp = max(x.get("cpu_procs", 0) for x in rows)
    res = "sim10k_1052x1914"
    bars = [
        ("Official, 1 core", kernel_best(rows, res, "official_cpu", 1), C["official"]),
        (f"Official, {maxp} cores", kernel_best(rows, res, "official_cpu", maxp), C["official"]),
        (f"FlashPASTA, {maxp} cores", kernel_best(rows, res, "ours_cpu", maxp), C["ours_cpu"]),
        ("FlashPASTA, 1 GPU", kernel_best(rows, res, "ours_gpu_rfft2"), C["gpu"]),
    ]
    base = bars[1][1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 5.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.55})
    for i, (lab, v, col) in enumerate(bars):
        a.barh(i, v, color=col, height=0.66)
        txt = f"{v:,.0f}" if v >= 10 else f"{v:.1f}"
        if v > base * 1.5:
            txt += f"  ({v / base:,.0f}×)"
        a.text(v * 1.15, i, txt, va="center", color=INK, fontsize=20)
    a.set_yticks(range(len(bars)), [x[0] for x in bars])
    a.invert_yaxis()
    a.set_xscale("log")
    a.set_xlim(0.3, max(x[1] for x in bars) * 30)
    a.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1k"])
    a.minorticks_off()
    a.set_xlabel("Images / s (log)")
    a.set_title("At Sim10k size, 1914×1052")
    a.grid(axis="y", visible=False)

    xs = list(range(len(RES_ORDER)))
    for variant, procs, col, lab in [("official_cpu", maxp, C["official"], f"Official, {maxp} cores"),
                                     ("ours_cpu", maxp, C["ours_cpu"], f"FlashPASTA, {maxp} cores"),
                                     ("ours_gpu_rfft2", None, C["gpu"], "FlashPASTA, 1 GPU")]:
        line(b, xs, [kernel_best(rows, r, variant, procs) for r in RES_ORDER], col, lab)
    b.set_yscale("log")
    b.set_xticks(xs, [RES_NAME[r] for r in RES_ORDER])
    b.set_xlabel("Image size")
    b.set_ylabel("Images / s (log)")
    b.set_title("Across image sizes")
    b.set_ylim(top=b.get_ylim()[1] * 30)
    b.legend(loc="upper right", fontsize=18, handlelength=1.6, borderaxespad=0.1)
    save(fig, "fig1_kernel")


# ---------------------------------------------------------------- training helpers
def load_train():
    out = []
    for p in sorted((RES / "train").glob("*.json")):
        d = json.loads(p.read_text())
        ep = d["epochs"][-1]  # last epoch: steady state
        out.append({**d["meta"], **ep, "file": p.name})
    return out


def demand(runs, imgsz):
    """Training throughput when data is never the bottleneck: best no-PASTA run."""
    r = [x["img_per_s_global"] for x in runs if x["pasta_mode"] == "none" and x["imgsz"] == imgsz and x["world_size"] == 1]
    return max(r) if r else None


# ---------------------------------------------------------------- loader
def fig_loader():
    path = RES / "loader.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    runs = load_train()
    by = defaultdict(list)
    for r in d["rows"]:
        by[(str(r["imgsz"]), r["mode"])].append((r["workers"], r["img_per_s"]))
    sizes = [s for s in ("640", "native") if any(k[0] == s for k in by)]
    fig, axes = plt.subplots(1, len(sizes), figsize=(COL_W, 5.6), gridspec_kw={"wspace": 0.3}, squeeze=False)
    for ax, sz in zip(axes[0], sizes):
        ws = sorted({w for (s, _), pts in by.items() if s == sz for w, _ in pts})
        for mode in MODES:
            pts = sorted(by.get((sz, mode), []))
            if pts:
                line(ax, [p[0] for p in pts], [p[1] for p in pts], MODE_COLOR[mode], LABEL[mode], dashed=mode == "none")
        ax.set_yscale("log")
        log2_workers(ax, ws)
        if sz == "640":
            dmd = demand(runs, 640)
            if dmd:
                ax.axhline(dmd, color=INK, lw=2, ls=":", zorder=2, label=f"YOLO26s trains at {dmd:,.0f}/s")
        ax.set_title("Resized to 640" if sz == "640" else "Native 1914×1052")
    axes[0][0].set_ylabel("Images / s delivered (log)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3 if len(handles) > 4 else 2,
               bbox_to_anchor=(0.5, 1.16), fontsize=18,
               handlelength=LEGEND_HANDLE, columnspacing=1.2)
    save(fig, "fig2_loader")


# ---------------------------------------------------------------- training
def fig_train():
    runs = [r for r in load_train() if r["world_size"] == 1 and r["imgsz"] == 640]
    if not runs:
        return
    ws = sorted({r["workers"] for r in runs})
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 5.6), gridspec_kw={"wspace": 0.3})
    for mode in MODES:
        pts = sorted((r["workers"], r["img_per_s_global"], r["data_wait_frac"]) for r in runs if r["pasta_mode"] == mode)
        if not pts:
            continue
        x = [p[0] for p in pts]
        line(a, x, [p[1] for p in pts], MODE_COLOR[mode], LABEL[mode], dashed=mode == "none")
        line(b, x, [100 * p[2] for p in pts], MODE_COLOR[mode], LABEL[mode], dashed=mode == "none")
    for ax in (a, b):
        log2_workers(ax, ws)
    a.set_ylabel("Training images / s")
    a.set_ylim(bottom=0)
    a.set_title("Training throughput")
    b.set_ylabel("GPU waiting for data (%)")
    b.set_ylim(0, 100)
    b.set_title("Data-loader stall")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16), fontsize=18,
               handlelength=LEGEND_HANDLE, columnspacing=1.2)
    save(fig, "fig3_train")


# ---------------------------------------------------------------- scaling
def fig_scaling():
    runs = load_train()
    multi = [r for r in runs if r["world_size"] > 1]
    if not multi:
        return
    imgsz, workers = multi[0]["imgsz"], multi[0]["workers"]
    fig, ax = plt.subplots(figsize=(COL_W / 2, 5.6))
    x, ticks = 0, []
    for mode in ["cpu_official", "gpu"]:
        for n in (1, 2):
            r = [r for r in runs if r["pasta_mode"] == mode and r["world_size"] == n and r["imgsz"] == imgsz
                 and r["workers"] == workers]
            if not r:
                continue
            v = r[0]["img_per_s_global"]
            ax.bar(x, v, color=MODE_COLOR[mode], width=0.7)
            ax.text(x, v * 1.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=20)
            ticks.append((x, f"{n} GPU{'s' if n > 1 else ''}"))
            x += 1
        x += 0.6
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylabel("Training images / s")
    ax.set_title(f"1 → 2 GPUs ({workers} workers/GPU)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_scaling")


if __name__ == "__main__":
    fig_kernel()
    fig_loader()
    fig_train()
    fig_scaling()
