"""
validate_dataset.py
===================
Pre-training dataset audit (SOP Section 1):
  - boxes within (0,1), positive w/h, not degenerate
  - class IDs in {0,1}
  - train/val splits + per-class counts (imbalance)
  - images missing labels / labels missing images
  - duplicate images (md5), corrupt images, tiny images (<32px)

Usage:
  python scripts/validate_dataset.py --data data/dataset.yaml
"""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:
    Image = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sha(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    issues = []

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        imgs = (sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTS) if img_dir.exists() else [])
        print(f"\n=== {split}: {len(imgs)} images ===")

        box_cls = Counter()
        img_cls = Counter()
        missing_lbl = lbl_no_img = oor = zero_area = full_box = 0
        corrupt = tiny = 0
        hashes = defaultdict(list)

        for im in imgs:
            hashes[sha(im)].append(im.name)
            if Image:
                try:
                    with Image.open(im) as imo:
                        w, h = imo.size
                        if min(w, h) < 32:
                            tiny += 1
                except Exception:
                    corrupt += 1
                    issues.append(f"{split}: corrupt image {im.name}")
            lbl = lbl_dir / (im.stem + ".txt")
            if not lbl.exists():
                missing_lbl += 1
                continue
            present = set()
            for ln in lbl.read_text().splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    issues.append(f"{split}: bad line in {lbl.name}: {ln}")
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                except Exception:
                    issues.append(f"{split}: nonnumeric in {lbl.name}: {ln}")
                    continue
                if cid not in (0, 1):
                    issues.append(f"{split}: bad class id {cid} in {lbl.name}")
                    continue
                box_cls[cid] += 1
                present.add(cid)
                if not (0 < xc < 1 and 0 < yc < 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    oor += 1
                    issues.append(f"{split}: out-of-range box in {lbl.name}: {ln}")
                if bw * bh < 1e-4:
                    zero_area += 1
                if bw > 0.99 and bh > 0.99:
                    full_box += 1
            for c in present:
                img_cls[c] += 1

        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                if not any((img_dir / (lbl.stem + e)).exists() for e in IMG_EXTS):
                    lbl_no_img += 1

        dups = {k: v for k, v in hashes.items() if len(v) > 1}
        tot = box_cls[0] + box_cls[1]
        print(f"  boxes/class (instances): {dict(box_cls)}")
        print(f"  images/class: drone={img_cls[0]} bird={img_cls[1]}")
        print(f"  missing labels: {missing_lbl} | labels w/o image: {lbl_no_img}")
        print(f"  out-of-range boxes: {oor} | zero-area: {zero_area} | full-image: {full_box}")
        print(f"  corrupt: {corrupt} | tiny(<32px): {tiny}")
        print(f"  duplicate groups: {len(dups)} (files: {sum(len(v) for v in dups.values())})")
        if tot:
            print(f"  imbalance: drone {100*box_cls[0]/tot:.1f}%  bird {100*box_cls[1]/tot:.1f}%")

    print("\n=== ISSUE SUMMARY (first 40) ===")
    for i in issues[:40]:
        print(" -", i)
    print(f"total issue entries: {len(issues)}")


if __name__ == "__main__":
    main()
