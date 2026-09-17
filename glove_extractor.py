"""Glove extractor: BGR or grayscale image in, 0/255 glove mask of the same size out."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PARAMS_FILE = Path(__file__).with_name("params.json")
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_params():
    return json.loads(PARAMS_FILE.read_text())


def _morph(m, op, k):
    k = int(k)
    return m if k <= 1 else cv2.morphologyEx(m, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _stretch(img, lo, hi):
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img.astype(np.float32) - a) / max(b - a, 1) * 255, 0, 255).astype(np.uint8)


def _fill_holes(m):
    m = (m > 0).astype(np.uint8) * 255
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(ff, pad, seed, 255)
    return cv2.bitwise_or(m, cv2.bitwise_not(ff))


def _select(m, p):
    """Largest component. If it is small and touches the bottom edge (a close-up), also keep the
    other bottom-touching components, so fingers that enter the frame separately are not dropped."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return m
    h, w = m.shape
    area = st[1:, cv2.CC_STAT_AREA]
    first = 1 + int(np.argmax(area))
    bottom = st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT] == h
    keep = [first]
    if area[first - 1] < p["extra_cc_frac"] * h * w and bottom[first]:
        keep += [i for i in 1 + np.argsort(-area, kind="stable") if i != first and bottom[i]][:int(p["extra_cc_max"])]
    return (np.isin(lab, keep) * 255).astype(np.uint8)


def _sever(m, k, p):
    """Erode, keep the glove component(s), dilate back: breaks thin links to background objects."""
    if k <= 1 or m.sum() == 0:
        return m
    return _fill_holes(_morph(_select(_morph(m, cv2.MORPH_ERODE, k), p), cv2.MORPH_DILATE, k))


def _glove_blob(img, p):
    x = cv2.GaussianBlur(_stretch(img, p["stretch_lo"], p["stretch_hi"]), (0, 0), p["blur_sigma"])
    m = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[1]
    m = _morph(_morph(m, cv2.MORPH_OPEN, p["open_k"]), cv2.MORPH_CLOSE, p["close_k"])
    return _sever(_fill_holes(_select(m, p)), p["mask_sever_k"], p)


def _edges(img, p):
    x = cv2.medianBlur(_stretch(img, p["stretch_lo"], p["stretch_hi"]), int(p["median_k"]))
    x = _stretch(np.clip((x.astype(np.float32) / 255) ** p["gamma"] * 255, 0, 255).astype(np.uint8),
                 p["stretch_lo"], p["stretch_hi"])
    b = cv2.GaussianBlur(x, (0, 0), p["canny_sigma"])
    hi = float(np.percentile(cv2.magnitude(cv2.Scharr(b, cv2.CV_32F, 1, 0), cv2.Scharr(b, cv2.CV_32F, 0, 1)),
                             p["canny_pct"]))
    return _morph(cv2.Canny(b, hi * p["canny_ratio"], hi), cv2.MORPH_DILATE, 2 * int(p["edge_thick"]) + 1)


def extract(image, **params):
    """Return the glove mask. Keyword arguments override the values in params.json."""
    p = load_params()
    unknown = set(params) - set(p)
    if unknown:
        raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
    p.update(params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    img = cv2.resize(gray, (int(w * p["scale"]), int(h * p["scale"])), interpolation=cv2.INTER_AREA)

    mask = _glove_blob(img, p)
    if mask.any():
        # Carve the edges out of the blob; keep the cut only if it removed a plausible share of it.
        cut = _select(_sever(cv2.bitwise_and(mask, cv2.bitwise_not(_edges(img, p))), p["cut_sever_k"], p), p)
        if cut.any():
            cut = _fill_holes(cut)
            removed = ((mask > 0) & (cut == 0)).sum() / max((mask > 0).sum(), 1)
            if p["guard_lo"] <= removed <= p["guard_hi"]:
                mask = cut
    # Grow the mask on purpose: keeping some background is harmless, cutting off a defect is not.
    mask = _morph(mask, cv2.MORPH_DILATE, 2 * int(p["dilate_px"]) + 1)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python glove_extractor.py INPUT OUTPUT   (image -> mask.png, or folder -> folder)")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        jobs = [(f, dst / f"{f.stem}.png") for f in sorted(src.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(src, dst)]
    for i, (f, out) in enumerate(jobs, 1):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {f}")
            continue
        if not cv2.imwrite(str(out), extract(img)):
            sys.exit(f"cannot write {out}")
        print(f"[{i}/{len(jobs)}] {out}")
