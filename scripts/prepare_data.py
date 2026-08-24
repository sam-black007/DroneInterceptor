"""
prepare_data.py
===============
Merge all downloaded drone + bird datasets into a single 2-class YOLO dataset:

    class 0 = drone
    class 1 = bird

Reads the source list from download_datasets.DATA_SOURCES, so any dataset you
enable there is picked up automatically. Handles:
  - YOLO labels: remap class index (drone subtypes -> 0, bird -> 1)
  - VOC/XML labels: convert bbox -> YOLO (all objects -> target class)
  - image hardlink + label rewrite
  - global 90/10 train/val split (stratified by class)

Output:
  data/processed/images/{train,val}/...
  data/processed/labels/{train,val}/...
  data/dataset.yaml
"""

import os
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from download_datasets import DATA_SOURCES, RAW

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

SEED = 42
VAL_FRACTION = 0.10
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_BOX_AREA = 1e-4  # drop boxes smaller than ~6x6px @640 (noise)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def remap_yolo(lbl_path: Path, target: int) -> str:
    lines = []
    for line in lbl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, xc, yc, bw, bh = (float(parts[0]), *map(float, parts[1:5]))
        except ValueError:
            continue
        if bw <= 0 or bh <= 0 or bw * bh < MIN_BOX_AREA:
            continue
        lines.append(f"{target} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + "\n" if lines else ""


def voc_to_yolo(xml_path: Path, target: int, img_dir: Path) -> str:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return ""
    sz = root.find("size")
    if sz is None:
        return ""
    try:
        W = int(sz.find("width").text)
        H = int(sz.find("height").text)
    except Exception:
        return ""
    if W <= 0 or H <= 0:
        return ""
    out = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
        except Exception:
            continue
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        if 0 < xc < 1 and 0 < yc < 1 and 0 < bw <= 1 and 0 < bh <= 1 \
                and bw * bh >= MIN_BOX_AREA:
            out.append(f"{target} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(out) + "\n" if out else ""


def resolve_root(base: Path) -> Path:
    # find <root>/images/<split> for split in train/val/valid/test
    for img_dir in base.rglob("images"):
        root = img_dir.parent
        if any((root / "images" / sp).exists()
               for sp in ("train", "val", "valid", "test")):
            return root
    for lbl_dir in base.rglob("labels"):
        root = lbl_dir.parent
        if any((root / "labels" / sp).exists()
               for sp in ("train", "val", "valid", "test")):
            return root
    return base


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    # recursive fallback
    for ext in IMG_EXTS:
        hits = list(img_dir.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def iter_pairs(source: dict, target: int):
    base = RAW / source["kind"] / source["name"]
    if not base.exists():
        return
    fmt = source["fmt"]

    if fmt == "voc":
        img_dir = base / "images"
        for xml in base.rglob("*.xml"):
            img = find_image(img_dir, xml.stem)
            if not img:
                continue
            text = voc_to_yolo(xml, target, img_dir)
            if text:
                yield img, text
        return

    # YOLO (handles layout variants:
    #   A: <root>/images/<split> + <root>/labels/<split>
    #   B: <root>/<split>/images + <root>/<split>/labels
    #   flat: <root>/images + <root>/labels ; <root> may have a wrapper subdir)
    root = base
    if not ((root / "images").exists() or (root / "train").exists()):
        subs = [d for d in base.iterdir()
                if d.is_dir() and not d.name.startswith(".")]
        if len(subs) == 1:
            root = subs[0]

    if (root / "images" / "train").exists() or (root / "images" / "valid").exists():
        layout = "img_under_root"
    elif (root / "train" / "images").exists() or (root / "valid" / "images").exists():
        layout = "split_under_root"
    else:
        layout = "flat"

    if layout == "flat":
        splits = [""]
    else:
        splits = [sp for sp in ("train", "val", "valid", "test")
                  if ((root / "images" / sp).exists()
                      if layout == "img_under_root"
                      else (root / sp / "images").exists())]

    for split in splits:
        if layout == "flat":
            img_dir = root / "images"
            lbl_dir = root / "labels"
        elif layout == "img_under_root":
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
        else:
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                text = remap_yolo(lbl, target)
                if text:
                    yield img, text


def main() -> None:
    random.seed(SEED)
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    PROCESSED.mkdir(parents=True)

    entries = []
    CLASS2IDX = {"drone": 0, "bird": 1}
    for src in DATA_SOURCES:
        tgt = CLASS2IDX[src["klass"]]
        count = 0
        for img_path, text in iter_pairs(src, tgt):
            # unique stem avoids collisions when sources reuse filenames
            uniq = f"{src['name']}__{img_path.stem}"
            h = hashlib.md5(img_path.read_bytes()).hexdigest()
            entries.append((img_path, text, tgt, uniq, img_path.suffix, h))
            count += 1
        status = "skipped (not downloaded)" if count == 0 else f"{count} images"
        print(f"[INFO] {src['name']} ({src['kind']}/{src['fmt']}): {status}")

    if not entries:
        raise SystemExit("No labeled images found - run download_datasets.py first.")

    by_class = {0: [], 1: []}
    for e in entries:
        by_class[e[2]].append(e)

    train_pairs, val_pairs = [], []
    for cls, lst in by_class.items():
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * VAL_FRACTION))
        val_pairs.extend(lst[:n_val])
        train_pairs.extend(lst[n_val:])

    # De-duplicate across splits: force identical images into the same split
    # (prevents train/val leakage that would inflate validation scores).
    train_hashes = {e[5] for e in train_pairs}
    fixed_val, seen_val = [], set()
    for v in val_pairs:
        if v[5] in train_hashes or v[5] in seen_val:
            train_pairs.append(v)        # move leak/dup out of val
        else:
            seen_val.add(v[5])
            fixed_val.append(v)
    val_pairs = fixed_val

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    def write(pairs, split):
        img_out = PROCESSED / "images" / split
        lbl_out = PROCESSED / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, text, _, uniq, ext, _h in pairs:
            link_or_copy(img_path, img_out / (uniq + ext))
            (lbl_out / (uniq + ".txt")).write_text(text)

    write(train_pairs, "train")
    write(val_pairs, "val")

    (ROOT / "data" / "dataset.yaml").write_text(
        f"path: {PROCESSED.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: ['drone', 'bird']\n"
    )

    td = sum(1 for p in train_pairs if p[2] == 0)
    vd = sum(1 for p in val_pairs if p[2] == 0)
    print("\n[DONE] merged dataset")
    print(f"  train: {len(train_pairs)}  (drone {td}, bird {len(train_pairs)-td})")
    print(f"  val:   {len(val_pairs)}  (drone {vd}, bird {len(val_pairs)-vd})")
    print(f"  wrote {ROOT / 'data' / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
