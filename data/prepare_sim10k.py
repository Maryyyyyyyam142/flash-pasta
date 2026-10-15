"""Convert Sim10k (Pascal VOC layout) to an Ultralytics YOLO dataset, car class only.

Expects the extracted archives repro_10k_images.tgz and repro_10k_annotations.tgz:
  <src>/VOC2012/JPEGImages/*.jpg  and  <src>/VOC2012/Annotations/*.xml
Images are symlinked, not copied. All 10k images go to train (the source domain in
the PASTA setup); --val N images are also listed as a small val split so
Ultralytics can run its end-of-training validation quickly.

Usage: python data/prepare_sim10k.py --src ~/Downloads/sim10k --dst data/sim10k
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def convert(xml_path):
    root = ET.parse(xml_path).getroot()
    W = float(root.find("size/width").text)
    H = float(root.find("size/height").text)
    lines = []
    for obj in root.iter("object"):
        if obj.find("name").text.strip().lower() != "car":
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, x1), min(W, x2)
        y1, y2 = max(0.0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/sim10k")
    ap.add_argument("--val", type=int, default=200)
    args = ap.parse_args()

    voc = next(Path(args.src).expanduser().rglob("JPEGImages")).parent
    dst = Path(args.dst).resolve()
    img_dir, lbl_dir = dst / "images/train", dst / "labels/train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    names = []
    for img in sorted((voc / "JPEGImages").glob("*.jpg")):
        xml = voc / "Annotations" / f"{img.stem}.xml"
        if not xml.exists():
            continue
        lines = convert(xml)
        link = img_dir / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        names.append(str(link))
        n_img += 1
        n_box += len(lines)

    random.Random(0).shuffle(names)
    (dst / "train.txt").write_text("\n".join(sorted(names)) + "\n")
    (dst / "val_small.txt").write_text("\n".join(sorted(names[: args.val])) + "\n")
    (dst.parent / "sim10k.yaml").write_text(
        f"path: {dst}\ntrain: train.txt\nval: val_small.txt\nnames:\n  0: car\n"
    )
    print(f"{n_img} images, {n_box} car boxes -> {dst}  (yaml: {dst.parent / 'sim10k.yaml'})")


if __name__ == "__main__":
    main()
