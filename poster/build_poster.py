"""Build the one-page A0 landscape poster (poster/poster.pdf) from results/*.json and figures/*.svg.

All numbers in the text are computed from the result files, so re-running the
benchmarks and this script keeps the poster consistent.

Usage: .venv/bin/python bench/plot.py && .venv/bin/python poster/build_poster.py
"""
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, POSTER = ROOT / "results", ROOT / "poster"
HEADLINE_WORKERS = 8  # CPU cores per GPU on a typical shared HPC GPU node


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def sci(v, digits=0):
    """3.3e-06 -> '3.3×10⁻⁶' with real superscripts."""
    mant, exp = f"{v:.{digits}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mant}×10{str(int(exp)).translate(sup)}"


def fmt(v, digits=0):
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def numbers():
    n = {}
    k = load("kernel.json")
    if k:
        rows = k["rows"]
        best = lambda res, var, procs=None: max(  # noqa: E731
            r["img_per_s"] for r in rows if r["res"] == res and r["variant"] == var and r["img_per_s"]
            and (procs is None or r.get("cpu_procs") == procs))
        maxp = max(r.get("cpu_procs", 0) for r in rows)
        s = "sim10k_1052x1914"
        n.update(maxp=maxp, k_off1=best(s, "official_cpu", 1), k_offN=best(s, "official_cpu", maxp),
                 k_oursN=best(s, "ours_cpu", maxp), k_ours1=best(s, "ours_cpu", 1), k_gpu=best(s, "ours_gpu_rfft2"),
                 k_gpu_fft2=best(s, "ours_gpu_fft2"), k640_off1=best("640x640", "official_cpu", 1),
                 k640_gpu=best("640x640", "ours_gpu_rfft2"), gpu_name=k["meta"]["gpu"], cpu_name=k["meta"]["cpu"])
        n["k_speedup"] = n["k_gpu"] / n["k_offN"]
        n["k_alg"] = n["k_ours1"] / n["k_off1"]
        n["k_rfft"] = n["k_gpu"] / n["k_gpu_fft2"]

    e = load("equivalence.json")
    if e:
        fp32 = [v for r in e["exact"] for name, v in r["variants"].items() if "fp32" in name]
        n["eq_max_err"] = max(v["max_abs_err"] for v in fp32)
        n["eq_uint8"] = max(v["uint8_pixels_differing_frac"] for v in fp32)
        n["eq_uint8_max"] = max(v["uint8_max_diff"] for v in fp32)
        nd = e["noise_distribution"]
        n["eq_std_lo"] = min(r["std_ratio_min"] for r in nd)
        n["eq_std_hi"] = max(r["std_ratio_max"] for r in nd)
        od = e["output_distribution"]
        n["eq_px_total"] = 3 * od["H"] * od["W"]
        n["eq_px_off"] = round(od["pixel_mean_abs_z_gt3_frac"] * n["eq_px_total"])

    ld = load("loader.json")
    if ld:
        get = lambda sz, mode, w: next((r["img_per_s"] for r in ld["rows"]  # noqa: E731
                                        if str(r["imgsz"]) == sz and r["mode"] == mode and r["workers"] == w), None)
        n["ld_decode1"] = get("640", "none", 1)
        n["ld_off1"] = get("640", "cpu_official", 1)
        none_w, gpu_w, off_w = (get("640", m, HEADLINE_WORKERS) for m in ("none", "gpu", "cpu_official"))
        if none_w and gpu_w and off_w:
            n["ld_gpu_frac"] = gpu_w / none_w
            n["ld_off_slow"] = none_w / off_w

    runs = []
    tdir = RES / "train"
    if tdir.exists():
        for p in sorted(tdir.glob("*.json")):
            d = json.loads(p.read_text())
            runs.append({**d["meta"], **d["epochs"][-1]})
    single = [r for r in runs if r["world_size"] == 1 and r["imgsz"] == 640]
    pick = lambda mode, w: next((r for r in single if r["pasta_mode"] == mode and r["workers"] == w), None)  # noqa: E731
    off, gpu, none = pick("cpu_official", HEADLINE_WORKERS), pick("gpu", HEADLINE_WORKERS), pick("none", HEADLINE_WORKERS)
    if off and gpu:
        n.update(t_off=off["img_per_s_global"], t_gpu=gpu["img_per_s_global"],
                 t_none=none["img_per_s_global"] if none else None,
                 t_off_stall=100 * off["data_wait_frac"], t_gpu_stall=100 * gpu["data_wait_frac"],
                 t_off_util=(off["gpu_util_mean"] or [None])[0], t_gpu_util=(gpu["gpu_util_mean"] or [None])[0])
        n["t_speedup"] = n["t_gpu"] / n["t_off"]
        n["t_epoch_off"] = 10000 / n["t_off"] / 60
        n["t_epoch_gpu"] = 10000 / n["t_gpu"] / 60
        n["t_100ep_off"] = 100 * 10000 / n["t_off"] / 3600
        n["t_100ep_gpu"] = 100 * 10000 / n["t_gpu"] / 3600
        n["t_gpu_pasta_ms"] = 1000 * gpu["gpu_pasta_s_rank0"] / gpu["batches"]
        offs = [r for r in single if r["pasta_mode"] == "cpu_official"]
        n["t_off_best"] = max(r["img_per_s_global"] for r in offs)
        n["t_off_best_w"] = max(offs, key=lambda r: r["img_per_s_global"])["workers"]
        demand = max(r["img_per_s_global"] for r in single if r["pasta_mode"] == "none")
        n["t_demand"] = demand
        ok = sorted(r["workers"] for r in single if r["pasta_mode"] == "gpu" and r["img_per_s_global"] >= 0.9 * demand)
        n["t_gpu_min_w"] = ok[0] if ok else None
    mj = []
    if (RES / "train_marjan").exists():
        for p in sorted((RES / "train_marjan").glob("*.json")):
            d = json.loads(p.read_text())
            mj.append({**d["meta"], **d["epochs"][-1]})
    mpick = lambda mode: next((r for r in mj if r["pasta_mode"] == mode and r["workers"] == HEADLINE_WORKERS), None)  # noqa: E731
    m_off, m_gpu = mpick("cpu_official"), mpick("gpu")
    m_train = ROOT / "data/marjan/train/images"
    n["m_images"] = len(list(m_train.iterdir())) if m_train.exists() else 9498
    if m_off and m_gpu:
        n.update(m_off=m_off["img_per_s_global"], m_gpu=m_gpu["img_per_s_global"],
                 m_off_stall=100 * m_off["data_wait_frac"], m_gpu_stall=100 * m_gpu["data_wait_frac"])
        n["m_speedup"] = n["m_gpu"] / n["m_off"]
        n["m_epoch_off"] = n["m_images"] / n["m_off"] / 60
        n["m_epoch_gpu"] = n["m_images"] / n["m_gpu"] / 60
    hi = [r for r in runs if r["world_size"] == 1 and r["imgsz"] == 1280]
    if hi:
        n["hi"] = {(r["pasta_mode"], r["workers"]): r for r in hi}
    multi = [r for r in runs if r["world_size"] > 1]
    n["has_scaling"] = bool(multi)
    return n


PIPELINE_SVG = """
<svg viewBox="0 0 1000 300" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans" font-size="25">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
  <text x="0" y="28" font-weight="700" fill="#0b0b0b">Official</text>
  <rect x="0" y="42" width="640" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="14" y="70" fill="#52514e" font-size="21">each CPU worker, one image at a time</text>
  <rect x="14" y="82" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="89" y="112" text-anchor="middle">decode</text>
  <rect x="182" y="82" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="257" y="112" text-anchor="middle">resize</text>
  <rect x="350" y="82" width="276" height="44" rx="6" fill="#fbe1d6" stroke="#eb6834" stroke-width="2.5"/>
  <text x="488" y="112" text-anchor="middle">PASTA: FFT, float64</text>
  <line x1="644" y1="104" x2="700" y2="104" stroke="#52514e" stroke-width="3" marker-end="url(#ar)"/>
  <rect x="706" y="42" width="290" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="720" y="70" fill="#52514e" font-size="21">GPU</text>
  <rect x="720" y="82" width="262" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="851" y="112" text-anchor="middle">train step</text>

  <text x="0" y="186" font-weight="700" fill="#0b0b0b">FlashPASTA</text>
  <rect x="0" y="200" width="470" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="14" y="228" fill="#52514e" font-size="21">each CPU worker</text>
  <rect x="14" y="240" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="89" y="270" text-anchor="middle">decode</text>
  <rect x="182" y="240" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="257" y="270" text-anchor="middle">resize</text>
  <line x1="474" y1="262" x2="530" y2="262" stroke="#52514e" stroke-width="3" marker-end="url(#ar)"/>
  <rect x="536" y="200" width="460" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="550" y="228" fill="#52514e" font-size="21">GPU, whole batch at once</text>
  <rect x="550" y="240" width="250" height="44" rx="6" fill="#d5e5f9" stroke="#2a78d6" stroke-width="2.5"/>
  <text x="675" y="270" text-anchor="middle">PASTA: batched rFFT</text>
  <rect x="814" y="240" width="168" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="898" y="270" text-anchor="middle">train step</text>
</svg>
"""


def build():
    n = numbers()
    a = json.loads((POSTER / "authors.json").read_text())
    people = a["authors"]
    affils = list(dict.fromkeys(x["affiliation"] for x in people))
    mark = (lambda x: f"<sup>{affils.index(x['affiliation']) + 1}</sup>") if len(affils) > 1 else (lambda x: "")
    names_html = ", ".join(f"{escape(x['name'])}{mark(x)}" for x in people)
    affil_html = " · ".join((f"<sup>{i + 1}</sup>" if len(affils) > 1 else "") + escape(af) for i, af in enumerate(affils))
    email_html = " · ".join(escape(x["email"]) for x in people)
    code_url = a.get("code_url", "")
    has_qr = code_url and (ROOT / "figures/qr_code.svg").exists()
    has_example = (ROOT / "figures/example_pasta.jpg").exists()
    has_train = "t_speedup" in n

    tiles = [
        (f"{fmt(n.get('k_speedup'))}×", f"FlashPASTA on 1 GPU vs official code on {n.get('maxp', 32)} CPU cores"),
        (f"{fmt(n.get('t_off_stall'))}% → {fmt(n.get('t_gpu_stall'), 1)}%" if has_train else "pending",
         f"of training time the GPU sits waiting for data ({HEADLINE_WORKERS} CPU workers)"),
        (f"{fmt(n.get('t_speedup'), 1)}×" if has_train else "pending",
         f"faster YOLO training on Sim10k with the same {HEADLINE_WORKERS} workers"),
        (sci(n["eq_max_err"]) if "eq_max_err" in n else "—",
         "largest pixel difference from the official output, same noise"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="big">{escape(b)}</div><div class="cap">{escape(c)}</div></div>'
                         for b, c in tiles)

    example_html = f"""
  <div class="example">
    <div><img src="../figures/example_original.jpg"><div class="lbl">Sim10k frame [2]</div></div>
    <div><img src="../figures/example_pasta.jpg"><div class="lbl">After PASTA (α=3, β=0.25, k=2)</div></div>
  </div>""" if has_example else ""

    hi = n.get("hi", {})
    hi_off, hi_gpu = hi.get(("cpu_official", HEADLINE_WORKERS)), hi.get(("gpu", HEADLINE_WORKERS))
    hi_note = (f" At 1280 px the gap widens to <b>{fmt(hi_gpu['img_per_s_global'] / hi_off['img_per_s_global'], 1)}×</b>."
               if hi_off and hi_gpu else "")
    if has_train:
        train_html = f"""
  <table class="res">
    <tr><th>{HEADLINE_WORKERS} CPU workers, 640 px</th><th class="off">Official, CPU</th><th class="gpu">FlashPASTA, GPU</th></tr>
    <tr><td>Training throughput</td><td>{fmt(n['t_off'])} img/s</td><td><b>{fmt(n['t_gpu'])} img/s</b></td></tr>
    <tr><td>GPU waiting for data</td><td>{fmt(n['t_off_stall'])}%</td><td><b>{fmt(n['t_gpu_stall'], 1)}%</b></td></tr>
    <tr><td>One Sim10k epoch (10k images)</td><td>{fmt(n['t_epoch_off'], 1)} min</td><td><b>{fmt(n['t_epoch_gpu'], 1)} min</b></td></tr>
    <tr><td>100 epochs (projected)</td><td>{fmt(n['t_100ep_off'], 1)} h</td><td><b>{fmt(n['t_100ep_gpu'], 1)} h</b></td></tr>
  </table>
  <p>The official version never feeds the GPU's {fmt(n['t_demand'])} img/s, not even with
  {n['t_off_best_w']} workers ({fmt(n['t_off_best'])} img/s).{hi_note}</p>"""
    else:
        train_html = "<p class='pending'>Training runs in progress.</p>"

    qr_html = f"""<div class="qr"><img src="../figures/qr_code.svg"><div>Code &amp; data</div></div>""" if has_qr else ""

    short_gpu = n.get("gpu_name", "").replace("NVIDIA ", "")
    short_cpu = re.sub(r"^AMD Ryzen |^Intel\(R\) ", "", n.get("cpu_name", ""))
    short_cpu = re.sub(r" (\d+)-Cores?$", r" (\1 cores)", short_cpu)
    if "m_speedup" in n:
        coral_rows = f"""
  <table class="res">
    <tr><th>{HEADLINE_WORKERS} CPU workers, 640 px</th><th class="off">Official, CPU</th><th class="gpu">FlashPASTA, GPU</th></tr>
    <tr><td>GPU waiting for data</td><td>{fmt(n['m_off_stall'])}%</td><td><b>{fmt(n['m_gpu_stall'], 1)}%</b></td></tr>
    <tr><td>One epoch ({fmt(n['m_images'])} images)</td><td>{fmt(n['m_epoch_off'], 1)} min</td><td><b>{fmt(n['m_epoch_gpu'], 1)} min</b></td></tr>
  </table>"""
        coral_head = f"4 · Red Sea corals: {fmt(n['m_speedup'], 1)}× faster training"
    else:
        coral_rows = "<p class='pending'>Coral runs in progress.</p>"
        coral_head = "4 · Red Sea coral reefs"
    thumbs = "".join(f'<div><img src="../figures/coral_{c}.jpg"><div class="lbl">{c.capitalize()}</div></div>'
                     for c in ("healthy", "bleached", "dead") if (ROOT / f"figures/coral_{c}.jpg").exists())
    coral_html = f"""<section><h2>{coral_head}</h2>
  <p>Real underwater images, Marjan dataset [4] ({fmt(n['m_images'])} train), same setup.</p>
  <div class="corals">{thumbs}</div>
  {coral_rows}</section>"""

    loader_claim = (f"GPU PASTA is nearly free" if n.get("ld_gpu_frac", 0) >= 0.9 else "PASTA on CPU vs GPU")
    dotted_note = ""
    train_claim = (f"YOLO training: {fmt(n['t_speedup'], 1)}× faster epochs" if has_train
                   else "YOLO training on Sim10k")

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FlashPASTA poster</title>
<style>
@page {{ size: 1189mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; }}
body {{ width: 1189mm; height: 841mm; position: relative; overflow: hidden; font-family: 'Noto Sans', sans-serif;
       font-size: 29pt; line-height: 1.32; color: #0b0b0b; background: #fff; }}
sup {{ line-height: 0; font-size: 0.7em; }}
b {{ font-weight: 700; }}
.lead {{ font-size: 29pt; font-weight: 600; color: #184f95; margin-bottom: 3mm; }}
header {{ background: #184f95; color: #fff; padding: 14mm 30mm 12mm; position: relative; }}
header .titles {{ margin-right: 330mm; }}
header .right {{ position: absolute; right: 30mm; top: 0; bottom: 0; display: flex; align-items: center; gap: 12mm; }}
.brand {{ color: #9ec5f4; }}
h1 {{ font-size: 80pt; line-height: 1.04; margin: 0 0 6mm; letter-spacing: -0.5pt; font-weight: 700; }}
.authors {{ font-size: 36pt; font-weight: 600; }}
.affil {{ font-size: 27pt; color: #cde2fb; }}
.conf {{ width: 240mm; text-align: right; font-size: 30pt; color: #cde2fb; line-height: 1.35; white-space: nowrap; }}
.conf b {{ color: #fff; font-size: 38pt; }}
.qr {{ background: #fff; color: #184f95; border-radius: 3mm; padding: 3mm 3mm 2mm; text-align: center;
       font-size: 20pt; font-weight: 600; }}
.qr img {{ width: 56mm; height: 56mm; display: block; }}
main {{ padding: 9mm 30mm 0; }}
.tiles {{ display: flex; gap: 10mm; margin-bottom: 9mm; }}
.tile {{ flex: 1; background: #f4f3f0; border-top: 2.5mm solid #2a78d6; border-radius: 0 0 3mm 3mm; padding: 5mm 7mm 6mm; }}
.tile .big {{ font-size: 68pt; font-weight: 700; color: #184f95; line-height: 1.05; }}
.tile .cap {{ font-size: 25pt; color: #52514e; line-height: 1.25; }}
.cols {{ display: flex; gap: 16mm; align-items: flex-start; }}
.col {{ flex: 1; }}
section {{ margin: 0 0 8mm; }}
/*BALANCE*/
.col.wide {{ flex: 1.08; }}
h2 {{ font-size: 42pt; color: #184f95; margin: 0 0 4mm; padding-left: 5mm; border-left: 3mm solid #2a78d6;
      line-height: 1.15; }}
p {{ margin: 0 0 3.5mm; }}
ul {{ margin: 0 0 3mm; padding-left: 9mm; }} li {{ margin-bottom: 2mm; }}
img.fig {{ width: 100%; display: block; margin: 1mm 0 2mm; }}
.caption {{ font-size: 25pt; color: #3d3c39; line-height: 1.25; margin: 0; }}
.caption b {{ color: #0b0b0b; }}
.example {{ display: flex; gap: 4mm; margin: 2mm 0 4mm; }}
.example > div {{ flex: 1; }}
.example img {{ width: 100%; display: block; border-radius: 2mm; }}
.example .lbl {{ font-size: 24pt; color: #52514e; margin-top: 1mm; }}
.corals {{ display: flex; gap: 4mm; margin: 2mm 0 3mm; }}
.corals > div {{ flex: 1; }}
.corals img {{ width: 100%; display: block; border-radius: 2mm; }}
.corals .lbl {{ font-size: 24pt; color: #52514e; margin-top: 1mm; text-align: center; }}
svg.pipe {{ width: 100%; display: block; margin: 1mm 0 4mm; }}
.eq {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 25pt; background: #f4f3f0; padding: 3mm 4mm;
       border-radius: 2mm; margin: 2mm 0 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 27pt; margin: 2mm 0 4mm; }}
th, td {{ text-align: left; padding: 2mm 3mm; border-bottom: 0.5mm solid #d6d5d0; vertical-align: top; }}
th {{ color: #52514e; font-weight: 600; }}
td.num {{ text-align: right; font-weight: 700; color: #184f95; white-space: nowrap; }}
table.res th.off {{ color: #c4501f; }} table.res th.gpu {{ color: #1c5cab; }}
.box {{ background: #eaf2fc; border-radius: 3mm; padding: 6mm 7mm 3mm; }}
.box h2 {{ border-left: none; padding-left: 0; }}
footer {{ position: absolute; left: 30mm; right: 30mm; bottom: 10mm; padding-top: 5mm;
          border-top: 0.6mm solid #d6d5d0; display: flex; gap: 20mm;
          font-size: 20pt; color: #52514e; line-height: 1.3; }}
footer .refs {{ flex: 1; }}
footer .link {{ text-align: right; white-space: nowrap; font-size: 22pt; color: #184f95; font-weight: 600; }}
.pending {{ color: #b45309; }}
</style></head><body>
<header>
  <div class="titles">
    <h1><span class="brand">FlashPASTA</span>: taking Fourier augmentation<br>off the CPU critical path</h1>
    <div class="authors">{names_html}</div>
    <div class="affil">{affil_html}</div>
    <div class="affil">{email_html}</div>
  </div>
  <div class="right">
    <div class="conf"><b>Saudi HPC/AI 2026</b><br>KAUST · 26–28 October<br>Methods that improve<br>computational efficiency</div>
    {qr_html}
  </div>
</header>
<main>
<div class="tiles">{tiles_html}</div>
<div class="cols">
<div class="col">
  <section><h2>The problem</h2>
  <p>Synthetic frames come with free labels, but detectors trained on them lose accuracy on real photos.
  <b>PASTA</b> [1] closes part of the gap by randomly perturbing each image's Fourier amplitude spectrum,
  more strongly at high frequencies.</p>
  {example_html}
  <p>The official code runs per image inside CPU data-loader workers, with float64 FFTs and the noise mask
  rebuilt on every call: <b>{fmt(n.get('k_off1'), 1)} images/s per core</b> at 1914×1052. At 640 px it slows a
  worker from <b>{fmt(n.get('ld_decode1'))} to {fmt(n.get('ld_off1'), 1)} images/s</b>.</p></section>
  <section><h2>FlashPASTA</h2>
  {PIPELINE_SVG.replace("<svg ", '<svg class="pipe" ', 1)}
  <ul>
    <li><b>Batched cuFFT on the GPU</b> after the batch is transferred</li>
    <li><b>Noise mask cached</b> per image size</li>
    <li><b>F·ε replaces |F|·ε·e<sup>iφ</sup></b>: no abs, angle or exp</li>
    <li><b>Half-size real FFT</b>, exact because only the real part is kept:</li>
  </ul>
  <div class="eq">Re IFFT(F·ε) = IRFFT(F·(ε(f)+ε(−f))/2)</div></section>
  <section><h2>Same output as the official code</h2>
  <table>
    <tr><td>Same noise: largest pixel difference (float32)</td><td class="num">{sci(n['eq_max_err'], 1) if 'eq_max_err' in n else '—'}</td></tr>
    <tr><td>8-bit pixels that change (by {fmt(n.get('eq_uint8_max'))} level)</td><td class="num">{100 * n.get('eq_uint8', 0):.3f}%</td></tr>
    <tr><td>Sampled noise spread vs official</td><td class="num">{fmt(n.get('eq_std_lo'), 2)}–{fmt(n.get('eq_std_hi'), 2)}×</td></tr>
    <tr><td>Pixel means that differ (|z| &gt; 3)</td><td class="num">{fmt(n.get('eq_px_off'))} / {fmt(n.get('eq_px_total'))}</td></tr>
  </table></section>
</div>
<div class="col wide">
  <section><h2>1 · PASTA alone: {fmt(n.get('k_speedup'))}× faster on one GPU</h2>
  <img class="fig" src="../figures/fig1_kernel.svg">
  <p class="caption"><b>Figure 1.</b> Augmentation only, no data loading. {escape(short_gpu)} vs
  {escape(short_cpu)}; CPU runs use single-threaded processes, like data-loader workers.</p></section>
  <section><h2>2 · Data loading: {loader_claim}</h2>
  <img class="fig" src="../figures/fig2_loader.svg">
  <p class="caption"><b>Figure 2.</b> Images/s a PyTorch DataLoader delivers on Sim10k (JPEG decode →
  letterbox → PASTA) vs CPU workers.{dotted_note}</p></section>
  <section class="box"><h2>Takeaways</h2><ul>
    <li>FlashPASTA on one GPU is {fmt(n.get('k_speedup'))}× faster than the official code on {n.get('maxp', 32)} CPU cores,
    with the same output up to floating-point rounding.</li>
    <li>It no longer eats the CPU-core budget, so a node feeds its GPU with a few workers.</li>
    <li>Restructuring ({fmt(n.get('k_alg'), 1)}× on one core) and the real-FFT identity ({fmt(n.get('k_rfft'), 1)}× on GPU)
    apply to any Fourier-domain augmentation.</li></ul></section>
</div>
<div class="col wide">
  <section><h2>3 · {train_claim}</h2>
  <img class="fig" src="../figures/fig3_train.svg">
  <p class="caption"><b>Figure 3.</b> YOLO26s, 640 px, batch 16, one RTX A5000, mixed precision; mosaic off so
  every mode sees identical inputs.</p>
  {train_html}</section>
{coral_html}
</div>
</div>
</main>
<footer>
  <div class="refs">
  [1] P. Chattopadhyay, K. Sarangmath, V. Vijaykumar, J. Hoffman. PASTA: Proportional Amplitude Spectrum Training
  Augmentation for Syn-to-Real Domain Generalization. ICCV 2023.
  [2] M. Johnson-Roberson et al. Driving in the Matrix. ICRA 2017. Sim10k, doi:10.7302/e1f1-3d97, CC BY-NC 4.0.
  [3] Ultralytics YOLO26.
  [4] A. Aldhahri, E. Saif, H. Ali, M. Alsayed, F. Alshareef. Harnessing Computer Vision and Deep Learning to Monitor
  Coral Reef Health. ETASR 15(4), 2025, doi:10.48084/etasr.10324; Marjan dataset, Roboflow Universe, CC BY 4.0.</div>
  {f'<div class="link">{escape(code_url.removeprefix("https://"))}</div>' if code_url else ""}
</footer>
</body></html>"""
    html = html.replace("/*BALANCE*/", balance_css(html))
    out = POSTER / "poster.html"
    out.write_text(html)
    doc = render(html)
    assert len(doc.pages) == 1, f"poster overflows to {len(doc.pages)} pages"
    doc.write_pdf(POSTER / "poster.pdf")
    print("wrote", POSTER / "poster.pdf")


def render(html):
    from weasyprint import HTML

    return HTML(string=html, base_url=str(POSTER)).render()


PX_PER_MM = 96 / 25.4


def balance_css(html, max_gap_mm=30, bottom_pad_mm=6):
    """Lay the page out once, then spread each column's spare height evenly between its sections,
    so all columns end at the same line above the footer (WeasyPrint's column flexbox is unreliable)."""
    page = render(html).pages[0]._page_box
    first = {}
    for box in page.descendants():
        el = getattr(box, "element", None)
        if el is not None and id(el) not in first and box.element_tag in {"div", "section", "footer"}:
            first[id(el)] = (el, box)
    cols = [(el, b) for el, b in first.values() if el.tag == "div" and "col" in (el.get("class") or "").split()]
    footer = next(b for el, b in first.values() if el.tag == "footer")
    limit = footer.position_y - bottom_pad_mm * PX_PER_MM
    rules = []
    for i, (col_el, col_box) in enumerate(sorted(cols, key=lambda c: c[1].position_x), start=1):
        secs = [first[id(ch)][1] for ch in col_el if ch.tag == "section" and id(ch) in first]
        if len(secs) < 2:
            continue
        bottom = max(b.position_y + b.margin_height() for b in secs)
        spare_mm = (limit - bottom) / PX_PER_MM
        gap = 8 + max(0.0, min(max_gap_mm - 8, spare_mm / (len(secs) - 1)))
        rules.append(f".col:nth-child({i}) > section {{ margin-bottom: {gap:.1f}mm; }}")
    return "\n".join(rules)


if __name__ == "__main__":
    build()
