"""make_subset.py - sample a small balanced subset for quick smoke training."""
import os
import shutil
import random
from pathlib import Path


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed"
DST = ROOT / "data" / "processed_small"

PER_CLASS_TRAIN = 600
VAL_FRAC = 0.1


def main():
    random.seed(7)
    for split in ("train", "val"):
        for cls, _name in ((0, "drone"), (1, "bird")):
            slbl = SRC / "labels" / split
            simg = SRC / "images" / split
            dlbl = DST / "labels" / split
            dimg = DST / "images" / split
            dlbl.mkdir(parents=True, exist_ok=True)
            dimg.mkdir(parents=True, exist_ok=True)

            entries = [
                l for l in slbl.glob("*.txt")
                if l.read_text().split()[0] == str(cls)
            ]
            random.shuffle(entries)
            n = max(1, int(PER_CLASS_TRAIN * VAL_FRAC)) if split == "val" else PER_CLASS_TRAIN
            for l in entries[:n]:
                link_or_copy(slbl / l.name, dlbl / l.name)
                cand = [p for p in simg.glob(l.stem + ".*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
                if cand:
                    link_or_copy(cand[0], dimg / cand[0].name)

    yaml_text = (
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: ['drone', 'bird']\n"
    )
    (ROOT / "data" / "dataset_small.yaml").write_text(yaml_text)
    for sp in ("train", "val"):
        print(sp, "imgs", len(list((DST / "images" / sp).glob("*"))))


if __name__ == "__main__":
    main()
