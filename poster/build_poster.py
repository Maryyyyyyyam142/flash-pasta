"""Build the one-page A0 landscape poster (poster/poster.pdf) from results/*.json and figures/*.svg.

All numbers in the text are computed from the result files, so re-running the
benchmarks and this script keeps the poster consistent.

Usage: .venv/bin/python bench/plot.py && .venv/bin/python poster/build_poster.py
"""
import json
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

    ld = load("loader.json")
    if ld:
        get = lambda sz, mode, w: next((r["img_per_s"] for r in ld["rows"]  # noqa: E731
                                        if str(r["imgsz"]) == sz and r["mode"] == mode and r["workers"] == w), None)
        n["ld_decode1"] = get("640", "none", 1)
        n["ld_off1"] = get("640", "cpu_official", 1)

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

  <text x="0" y="186" font-weight="700" fill="#0b0b0b">Ours</text>
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
    has_train = "t_speedup" in n

    tiles = [
        (f"{fmt(n.get('k_speedup'))}×", f"faster PASTA on 1 GPU than the official code on {n.get('maxp', 32)} CPU cores"),
        (f"{fmt(n.get('t_off_stall'))}% → {fmt(n.get('t_gpu_stall'), 1)}%" if has_train else "pending",
         f"of training time the GPU waits for data ({HEADLINE_WORKERS} CPU workers)"),
        (f"{fmt(n.get('t_speedup'), 1)}×" if has_train else "pending",
         f"faster YOLO training epoch on Sim10k ({HEADLINE_WORKERS} workers)"),
        (sci(n["eq_max_err"]) if "eq_max_err" in n else "—",
         "largest pixel difference from the official output (same noise)"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="big">{escape(b)}</div><div class="cap">{escape(c)}</div></div>'
                         for b, c in tiles)

    if has_train:
        train_text = f"""
        <p>With {HEADLINE_WORKERS} CPU workers per GPU, a common allocation on shared GPU nodes,
        the official PASTA keeps the GPU waiting for data <b>{fmt(n['t_off_stall'])}%</b> of the time
        and training runs at <b>{fmt(n['t_off'])} img/s</b>. On the GPU, PASTA adds
        {fmt(n['t_gpu_pasta_ms'], 1)} ms per batch, the stall drops to <b>{fmt(n['t_gpu_stall'], 1)}%</b> and training
        runs at <b>{fmt(n['t_gpu'])} img/s</b>{f" (no PASTA: {fmt(n['t_none'])} img/s)" if n.get('t_none') else ""}.</p>
        <table>
          <tr><th></th><th>Official, CPU</th><th>Ours, GPU</th></tr>
          <tr><td>One Sim10k epoch (10k images)</td><td>{fmt(n['t_epoch_off'], 1)} min</td><td>{fmt(n['t_epoch_gpu'], 1)} min</td></tr>
          <tr><td>100 epochs (projected)</td><td>{fmt(n['t_100ep_off'], 1)} h</td><td>{fmt(n['t_100ep_gpu'], 1)} h</td></tr>
        </table>
        <p>Even with all {n['t_off_best_w']} workers the official version reaches only
        {fmt(n['t_off_best'])} img/s, below the {fmt(n['t_demand'])} img/s the GPU can train at.
        The GPU version gets within 10% of that rate with
        {n['t_gpu_min_w'] if n.get('t_gpu_min_w') else '—'} workers.</p>"""
    else:
        train_text = "<p class='pending'>Training runs in progress.</p>"

    takeaways = f"""
      <li>Moving PASTA from CPU workers to the GPU makes it {fmt(n.get('k_speedup'))}× faster than 32 CPU cores,
      with output identical to the official code up to floating-point rounding.</li>
      <li>The augmentation no longer consumes the CPU-core budget, so the same node can feed its GPU
      with a few workers.</li>
      <li>Restructuring alone ({fmt(n.get('k_alg'), 1)}× on one CPU core) and the real-FFT identity
      ({fmt(n.get('k_rfft'), 1)}× on GPU) carry over to any Fourier-domain augmentation.</li>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>GPU-accelerated PASTA</title>
<style>
@page {{ size: 1189mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; width: 1189mm; height: 841mm; padding: 24mm 30mm 20mm; font-family: 'Noto Sans', sans-serif;
       font-size: 27pt; line-height: 1.32; color: #0b0b0b; background: #fff; }}
header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 3mm solid #2a78d6;
         padding-bottom: 8mm; }}
h1 {{ font-size: 80pt; line-height: 1.05; margin: 0 0 6mm; letter-spacing: -0.5pt; }}
.authors {{ font-size: 36pt; }} .affil {{ font-size: 30pt; color: #52514e; }}
header > div:first-child {{ flex: 1; }}
.conf {{ flex: 0 0 230mm; text-align: right; font-size: 30pt; color: #52514e; }} .conf b {{ color: #0b0b0b; }}
.tiles {{ display: flex; gap: 10mm; margin: 9mm 0 8mm; }}
.tile {{ flex: 1; background: #f4f3f0; border-radius: 4mm; padding: 6mm 8mm; }}
.tile .big {{ font-size: 64pt; font-weight: 700; color: #1c5cab; line-height: 1.05; }}
.tile .cap {{ font-size: 25pt; color: #52514e; }}
.cols {{ display: flex; gap: 16mm; }}
.col {{ flex: 1; }} .col.wide {{ flex: 1.12; }}
h2 {{ font-size: 40pt; color: #1c5cab; margin: 0 0 3mm; padding-bottom: 2mm; border-bottom: 0.8mm solid #d6d5d0; }}
section {{ margin-bottom: 8mm; }}
p {{ margin: 0 0 4mm; }} ul, ol {{ margin: 0 0 4mm; padding-left: 10mm; }} li {{ margin-bottom: 2mm; }}
img.fig {{ width: 100%; display: block; margin: 2mm 0 2mm; }}
.caption {{ font-size: 24pt; color: #52514e; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 5mm; font-size: 26pt; }}
th, td {{ text-align: left; padding: 2mm 3mm; border-bottom: 0.5mm solid #d6d5d0; }}
th {{ color: #52514e; font-weight: 600; }}
.eq {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 24pt; background: #f4f3f0; padding: 3mm 4mm;
       border-radius: 2mm; margin: 2mm 0 4mm; }}
.refs {{ font-size: 21pt; color: #52514e; }}
.pending {{ color: #b45309; }}
</style></head><body>
<header>
  <div>
    <h1>PASTA at GPU speed: taking Fourier augmentation<br>off the CPU critical path</h1>
    <div class="authors">{escape(a['authors'])}</div>
    <div class="affil">{escape(a['affiliation'])} · {escape(a['email'])}</div>
  </div>
  <div class="conf"><b>Saudi HPC/AI 2026</b><br>KAUST, 26–28 October<br>Methods that improve<br>computational efficiency</div>
</header>
<div class="tiles">{tiles_html}</div>
<div class="cols">
<div class="col">
  <section><h2>Problem</h2>
  <p>Synthetic images from games and simulators come with free labels, but detectors trained on them
  lose accuracy on real photos. <b>PASTA</b> [1] narrows this gap by randomly perturbing each image's
  Fourier amplitude spectrum, more strongly at high frequencies.</p>
  <p>The official code runs inside CPU data-loader workers, one image at a time: two full FFTs in
  float64 and a frequency mask rebuilt on every call. At Sim10k's 1914×1052 that is
  <b>{fmt(n.get('k_off1'), 1)} images/s per CPU core</b>. At 640 px a worker that decodes and resizes
  {fmt(n.get('ld_decode1'))} images/s drops to {fmt(n.get('ld_off1'), 1)} with PASTA.</p></section>
  <section><h2>Approach</h2>
  {PIPELINE_SVG}
  <ol>
    <li><b>Batched on the GPU</b>: one cuFFT call for the whole batch, after it reaches the GPU.</li>
    <li><b>Cached mask</b>: the per-frequency noise level is built once per image size.</li>
    <li><b>No amplitude/phase split</b>: |F|·ε·e<sup>iφ</sup> = F·ε, so abs, angle and exp are skipped.</li>
    <li><b>Real FFT</b>: the official code keeps only the real part of the inverse FFT, which equals</li>
  </ol>
  <div class="eq">Re IFFT(F·ε) = IRFFT( F · (ε(f) + ε(−f)) / 2 )</div>
  <p>so half the spectrum and half the random numbers suffice, with noise drawn at the matching variance.</p></section>
  <section><h2>Correctness</h2>
  <p>Fed the same random draw as the official code, our output differs by at most
  <b>{sci(n['eq_max_err'], 1)}</b> (float32 GPU vs official); after rounding to 8-bit, at most
  {100 * n.get('eq_uint8', 0):.3f}% of pixels change, by {fmt(n.get('eq_uint8_max'))} level.
  With our own sampling, per-frequency noise spread matches within
  {fmt(n.get('eq_std_lo'), 2)}–{fmt(n.get('eq_std_hi'), 2)}× and per-pixel output statistics match.</p></section>
</div>
<div class="col wide">
  <section><h2>1 · PASTA alone</h2>
  <img class="fig" src="../figures/fig1_kernel.svg">
  <p class="caption">Images/s of the augmentation only. {escape(n.get('gpu_name', ''))} vs
  {escape(n.get('cpu_name', ''))}; CPU numbers use single-threaded processes, like data-loader workers.
  Speed-ups are relative to the official code on {n.get('maxp', 32)} cores.</p></section>
  <section><h2>2 · Input pipeline</h2>
  <img class="fig" src="../figures/fig2_loader.svg">
  <p class="caption">Images/s a PyTorch DataLoader delivers on Sim10k (JPEG decode → letterbox → PASTA),
  by number of CPU workers. Dotted line: rate at which YOLO26s trains at 640 px when data is never late.</p></section>
</div>
<div class="col wide">
  <section><h2>3 · YOLO training on Sim10k</h2>
  <img class="fig" src="../figures/fig3_train.svg">
  <p class="caption">YOLO26s, 640 px, batch 16, one RTX A5000, mixed precision; mosaic off so all modes see
  identical inputs. Steady-state epoch on 2,000 Sim10k images.</p>
  {train_text}</section>
  <section><h2>Takeaways</h2><ul>{takeaways}</ul></section>
  <section class="refs">
  [1] Chattopadhyay, Sarangmath, Vijaykumar, Hoffman. PASTA: Proportional Amplitude Spectrum Training
  Augmentation for Syn-to-Real Domain Generalization. ICCV 2023.
  [2] Johnson-Roberson et al. Driving in the Matrix. ICRA 2017; Sim10k via UMich Deep Blue,
  doi:10.7302/e1f1-3d97 (CC BY-NC 4.0). [3] Ultralytics YOLO26.
  {f"Code: {escape(a['code_url'])}" if a.get('code_url') else ""}</section>
</div>
</div>
</body></html>"""
    out = POSTER / "poster.html"
    out.write_text(html)
    from weasyprint import HTML

    HTML(filename=str(out), base_url=str(POSTER)).write_pdf(POSTER / "poster.pdf")
    print("wrote", POSTER / "poster.pdf")


if __name__ == "__main__":
    build()
