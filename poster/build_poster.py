"""Build the one-page A0 landscape poster (poster/poster.pdf) from results/*.json and figures/*.svg.

All numbers in the text are computed from the result files, so re-running the
benchmarks and this script keeps the poster consistent.

Usage: .venv/bin/python bench/plot.py && .venv/bin/python poster/build_poster.py
"""
import json
import math
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, POSTER = ROOT / "results", ROOT / "poster"
HEADLINE_WORKERS = 8  # CPU cores per GPU on a typical shared HPC GPU node
RES_METHODS = [("off", "off", "Official, CPU"), ("ours", "cpu", "FlashPASTA, CPU"), ("gpu", "gpu", "FlashPASTA, GPU")]


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


def fmt_max(v, digits=1):
    """Upper bound shown as '≤ v': round up, so 3.04 prints as 3.1 rather than 3.0."""
    return "—" if v is None else fmt(math.ceil(v * 10 ** digits) / 10 ** digits, digits)


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
                 k_ours1=best(s, "ours_cpu", 1), k_gpu=best(s, "ours_gpu_rfft2"), k_gpu_fft2=best(s, "ours_gpu_fft2"),
                 gpu_name=k["meta"]["gpu"], cpu_name=k["meta"]["cpu"])
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
        n["eq_px_draws"] = od["draws"]
        n["eq_px_std"] = od["pixel_std_ratio_median"]

    ld = load("loader.json")
    if ld:
        get = lambda sz, mode, w: next((r["img_per_s"] for r in ld["rows"]  # noqa: E731
                                        if str(r["imgsz"]) == sz and r["mode"] == mode and r["workers"] == w), None)
        n["ld_decode1"] = get("640", "none", 1)
        n["ld_off1"] = get("640", "cpu_official", 1)
        none_w, gpu_w = get("640", "none", HEADLINE_WORKERS), get("640", "gpu", HEADLINE_WORKERS)
        if none_w and gpu_w:
            n["ld_gpu_frac"] = gpu_w / none_w

    runs = []
    tdir = RES / "train"
    if tdir.exists():
        for p in sorted(tdir.glob("*.json")):
            d = json.loads(p.read_text())
            runs.append({**d["meta"], **d["epochs"][-1]})
    single = [r for r in runs if r["world_size"] == 1 and r["imgsz"] == 640]
    pick = lambda mode, w: next((r for r in single if r["pasta_mode"] == mode and r["workers"] == w), None)  # noqa: E731
    off, gpu, ours = (pick(m, HEADLINE_WORKERS) for m in ("cpu_official", "gpu", "cpu_ours"))
    if off and gpu and ours:
        n.update(t_off=off["img_per_s_global"], t_gpu=gpu["img_per_s_global"], t_ours=ours["img_per_s_global"],
                 t_off_stall=100 * off["data_wait_frac"], t_gpu_stall=100 * gpu["data_wait_frac"],
                 t_ours_stall=100 * ours["data_wait_frac"])
        n["t_speedup"] = n["t_gpu"] / n["t_off"]
        for m in ("off", "ours", "gpu"):
            n[f"t_epoch_{m}"] = 10000 / n[f"t_{m}"] / 60
        n["t_off_max_w"] = max(r["workers"] for r in single if r["pasta_mode"] == "cpu_official")
        off_max, none_max = pick("cpu_official", n["t_off_max_w"]), pick("none", n["t_off_max_w"])
        n.update(t_off_max=off_max["img_per_s_global"], t_off_max_stall=100 * off_max["data_wait_frac"])
        # Busy CPU workers can also slow the training process itself: compare time per step with no PASTA
        step = lambda r: r["step_s"] / r["batches"]  # noqa: E731
        n["t_off_max_step"] = 100 * (step(off_max) / step(none_max) - 1) if none_max else None
        n["t_demand"] = max(r["img_per_s_global"] for r in single if r["pasta_mode"] == "none")
        # Compare each PASTA mode with no PASTA at the same worker count: GPU PASTA's worst-case cost,
        # and the fewest workers with which FlashPASTA in CPU workers keeps up (within 5%)
        base = {r["workers"]: r["img_per_s_global"] for r in single if r["pasta_mode"] == "none"}
        rel = lambda mode: {r["workers"]: r["img_per_s_global"] / base[r["workers"]]  # noqa: E731
                            for r in single if r["pasta_mode"] == mode and r["workers"] in base}
        n["t_gpu_cost"] = 100 * (1 - min(rel("gpu").values()))
        n["t_ours_min_w"] = min((w for w, v in rel("cpu_ours").items() if v >= 0.95), default=None)
    mj = []
    if (RES / "train_marjan").exists():
        for p in sorted((RES / "train_marjan").glob("*.json")):
            d = json.loads(p.read_text())
            mj.append({**d["meta"], **d["epochs"][-1]})
    mpick = lambda mode: next((r for r in mj if r["pasta_mode"] == mode and r["workers"] == HEADLINE_WORKERS), None)  # noqa: E731
    m_off, m_gpu, m_ours = mpick("cpu_official"), mpick("gpu"), mpick("cpu_ours")
    m_train = ROOT / "data/marjan/train/images"
    n["m_images"] = len(list(m_train.iterdir())) if m_train.exists() else 9498
    if m_off and m_gpu and m_ours:
        n.update(m_off=m_off["img_per_s_global"], m_gpu=m_gpu["img_per_s_global"], m_ours=m_ours["img_per_s_global"],
                 m_off_stall=100 * m_off["data_wait_frac"], m_gpu_stall=100 * m_gpu["data_wait_frac"],
                 m_ours_stall=100 * m_ours["data_wait_frac"])
        n["m_speedup"] = n["m_gpu"] / n["m_off"]
        for m in ("off", "ours", "gpu"):
            n[f"m_epoch_{m}"] = n["m_images"] / n[f"m_{m}"] / 60
    hi = [r for r in runs if r["world_size"] == 1 and r["imgsz"] == 1280]
    if hi:
        n["hi"] = {(r["pasta_mode"], r["workers"]): r for r in hi}
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
  <rect x="0" y="200" width="346" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="14" y="228" fill="#52514e" font-size="21">each CPU worker</text>
  <rect x="14" y="240" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="89" y="270" text-anchor="middle">decode</text>
  <rect x="182" y="240" width="150" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="257" y="270" text-anchor="middle">resize</text>
  <line x1="350" y1="262" x2="406" y2="262" stroke="#52514e" stroke-width="3" marker-end="url(#ar)"/>
  <rect x="412" y="200" width="584" height="96" rx="10" fill="#f4f3f0" stroke="#d6d5d0"/>
  <text x="426" y="228" fill="#52514e" font-size="21">GPU, whole batch at once</text>
  <rect x="426" y="240" width="280" height="44" rx="6" fill="#d5e5f9" stroke="#2a78d6" stroke-width="2.5"/>
  <text x="566" y="270" text-anchor="middle">PASTA: rFFT, float32</text>
  <rect x="720" y="240" width="262" height="44" rx="6" fill="#fff" stroke="#bdbcb6"/><text x="851" y="270" text-anchor="middle">train step</text>
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
        (sci(n["eq_max_err"], 1) if "eq_max_err" in n else "—",
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
    hi_note = (f" At 1280 px: <b>{fmt(hi_gpu['img_per_s_global'] / hi_off['img_per_s_global'], 1)}×</b> (GPU vs official)."
               if hi_off and hi_gpu else "")

    def res_table(p, epoch_head):
        """One row per method, coloured like the figures; p is the key prefix ('t' Sim10k, 'm' Marjan)."""
        rows = "".join(
            f'<tr><th class="{cls}">{label}</th>'
            + "".join(f"<td>{v}</td>" if m == "off" else f"<td><b>{v}</b></td>" for v in (
                fmt(n[f"{p}_{m}"]), f"{fmt(n[f'{p}_{m}_stall'], 0 if m == 'off' else 1)}%",
                f"{fmt(n[f'{p}_epoch_{m}'], 1)} min"))
            + "</tr>"
            for m, cls, label in RES_METHODS)
        return (f'<table class="res"><tr><th>{HEADLINE_WORKERS} CPU workers, 640 px</th><th>Images/s</th>'
                f"<th>GPU waiting</th><th>{epoch_head}</th></tr>{rows}</table>")

    if has_train:
        if n["t_off_max_stall"] < 5 and n["t_off_max_step"]:  # loader keeps up, but the workers slow each step
            off_note = (f": {n['t_off_max_w']} workers reach {fmt(n['t_off_max'])} img/s but compete with training for the CPU,"
                        f" slowing each step by {fmt(n['t_off_max_step'])}%")
        else:
            off_note = f", even with {n['t_off_max_w']} workers"
        train_html = f"""
  {res_table("t", "Sim10k epoch")}
  <p>The official code never keeps up with the GPU's {fmt(n['t_demand'])} img/s{off_note}.{hi_note}</p>"""
    else:
        train_html = "<p class='pending'>Training runs in progress.</p>"

    # The QR code is also a link, for readers of the PDF
    qr_html = (f'<div class="qr"><a href="{escape(code_url)}"><img src="../figures/qr_code.svg"></a>'
               f"<div>Code &amp; data</div></div>") if has_qr else ""

    short_gpu = n.get("gpu_name", "").replace("NVIDIA ", "")
    short_cpu = re.sub(r"^AMD Ryzen |^Intel\(R\) ", "", n.get("cpu_name", ""))
    short_cpu = re.sub(r" (\d+)-Cores?$", r" (\1 cores)", short_cpu)
    if "m_speedup" in n:
        coral_rows = res_table("m", "Epoch")
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

    loader_claim = "GPU PASTA is nearly free" if n.get("ld_gpu_frac", 0) >= 0.9 else "PASTA on CPU vs GPU"
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
.qr a, .qr img {{ display: block; }}
.qr img {{ width: 56mm; height: 56mm; }}
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
table.res th.off {{ color: #c4501f; }} table.res th.cpu {{ color: #12805a; }} table.res th.gpu {{ color: #1c5cab; }}
.box {{ background: #eaf2fc; border-radius: 3mm; padding: 6mm 7mm 3mm; }}
.box h2 {{ border-left: none; padding-left: 0; }}
footer {{ position: absolute; left: 30mm; right: 30mm; bottom: 10mm; padding-top: 5mm;
          border-top: 0.6mm solid #d6d5d0; display: flex; gap: 20mm;
          font-size: 20pt; color: #52514e; line-height: 1.3; }}
footer .refs {{ flex: 1; }}
.nw {{ white-space: nowrap; }}
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
  <p>The official code runs per image inside CPU data-loader workers, with float64 noise and inverse FFT and
  the noise mask rebuilt on every call: <b>{fmt(n.get('k_off1'), 1)} images/s per core</b> at 1914×1052. At 640 px it slows a
  worker from <b>{fmt(n.get('ld_decode1'))} to {fmt(n.get('ld_off1'), 1)} images/s</b>.</p></section>
  <section><h2>FlashPASTA</h2>
  {PIPELINE_SVG.replace("<svg ", '<svg class="pipe" ', 1)}
  <ul>
    <li><b>Batched cuFFT on the GPU</b> after the batch is transferred</li>
    <li><b>Noise mask cached</b> per image size</li>
    <li><b>float32 throughout</b>{f", within {sci(n['eq_max_err'], 1)} of the official output" if 'eq_max_err' in n else ""}</li>
    <li><b>F·ε replaces |F|·ε·e<sup>iφ</sup></b>: no abs, angle or exp</li>
    <li><b>Half-size real FFT</b>, exact because only the real part is kept:</li>
  </ul>
  <div class="eq">Re IFFT(F·ε) = IRFFT(F·(ε(f)+ε(−f))/2)</div></section>
  <section><h2>Same output as the official code</h2>
  <table>
    <tr><td>Same noise: largest pixel difference (float32)</td><td class="num">{sci(n['eq_max_err'], 1) if 'eq_max_err' in n else '—'}</td></tr>
    <tr><td>8-bit pixels that change (by {fmt(n.get('eq_uint8_max'))} level)</td><td class="num">{100 * n.get('eq_uint8', 0):.3f}%</td></tr>
    <tr><td>Sampled noise spread vs official</td><td class="num">{fmt(n.get('eq_std_lo'), 2)}–{fmt(n.get('eq_std_hi'), 2)}×</td></tr>
    <tr><td>Output pixel spread vs official ({fmt(n.get('eq_px_draws'))} draws, median)</td><td class="num">{fmt(n.get('eq_px_std'), 2)}×</td></tr>
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
  letterbox → PASTA) vs CPU workers.</p></section>
  <section class="box"><h2>Takeaways</h2><ul>
    <li>FlashPASTA on one GPU is {fmt(n.get('k_speedup'))}× faster than the official code on {n.get('maxp', 32)} CPU cores,
    with the same output up to floating-point rounding.</li>
    <li>On the GPU it costs ≤{fmt_max(n.get('t_gpu_cost'))}% of training speed with any number of workers;
    in CPU workers FlashPASTA needs {n.get('t_ours_min_w', '—')} to keep up.</li>
    <li>Restructuring ({fmt(n.get('k_alg'), 1)}× on one core) and the real-FFT identity ({fmt(n.get('k_rfft'), 1)}× on GPU)
    apply to any Fourier-domain augmentation.</li></ul></section>
</div>
<div class="col wide">
  <section><h2>3 · {train_claim}</h2>
  <img class="fig" src="../figures/fig3_train.svg">
  <p class="caption"><b>Figure 3.</b> YOLO26s [3], 640 px, batch 16, one RTX A5000, mixed precision; mosaic off so
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
  [2] M. Johnson-Roberson et al. Driving in the Matrix. ICRA 2017. Sim10k, doi:10.7302/e1f1-3d97,
  <span class="nw">CC BY-NC 4.0</span>.
  [3] Ultralytics YOLO26.
  [4] A. Aldhahri, E. Saif, H. Ali, M. Alsayed, F. Alshareef. Harnessing Computer Vision and Deep Learning to Monitor
  Coral Reef Health. ETASR 15(4), 2025, doi:10.48084/etasr.10324; Marjan dataset, Roboflow Universe,
  <span class="nw">CC BY 4.0</span>.</div>
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
        # The page count can't catch this: the body has a fixed height, so a long column runs under the footer
        overrun_mm = (max(b.position_y + b.margin_height() - b.margin_bottom for b in secs) - footer.position_y) / PX_PER_MM
        assert overrun_mm <= 0, f"column {i} runs {overrun_mm:.0f} mm into the footer"
        if len(secs) < 2:
            continue
        bottom = max(b.position_y + b.margin_height() for b in secs)
        spare_mm = (limit - bottom) / PX_PER_MM
        gap = 8 + max(0.0, min(max_gap_mm - 8, spare_mm / (len(secs) - 1)))
        rules.append(f".col:nth-child({i}) > section {{ margin-bottom: {gap:.1f}mm; }}")
    return "\n".join(rules)


if __name__ == "__main__":
    build()
