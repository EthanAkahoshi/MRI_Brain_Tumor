# Per-image statistics (no model) for every file in the training pool, Testing
# and the external set, plus 64x64 grayscale thumbnails for near-duplicate search.
# Output: image_stats.csv, thumbs.npz (in this folder)
import csv, os
import cv2
import numpy as np
from common import pool_files, benchmark_files, external_files, file_hash, rel
from model import BACKGROUND_THRESHOLD, crop_to_brain

OUT = os.path.dirname(os.path.abspath(__file__))
HIST_BINS = 16
rows, raw_thumbs, crop_thumbs = [], [], []

for path, label, source in pool_files() + benchmark_files() + external_files():
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    b, g, r = [bgr[:, :, i].astype(np.int16) for i in range(3)]
    chan_diff = max(np.abs(b - g).mean(), np.abs(g - r).mean(), np.abs(b - r).mean())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > BACKGROUND_THRESHOLD).astype(np.uint8)
    crop = crop_to_brain(bgr)
    ch, cw = crop.shape[:2]
    cgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    n, cc, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    big = st[1 + areas.argmax()] if n > 1 else None
    largest_bbox = big[2] * big[3] if big is not None else 0
    k = max(2, int(0.08 * min(h, w)))
    corners = np.concatenate([gray[:k, :k].ravel(), gray[:k, -k:].ravel(), gray[-k:, :k].ravel(), gray[-k:, -k:].ravel()])
    hist = np.histogram(gray, bins=HIST_BINS, range=(0, 256))[0] / gray.size
    rows.append(dict(
        path=rel(path) if source != 'external' else path, md5=file_hash(path), source=source, label=label,
        ext=os.path.splitext(path)[1].lower(), bytes=os.path.getsize(path), height=h, width=w,
        aspect=w / h, chan_diff=chan_diff, mean=gray.mean(), std=gray.std(),
        p99=np.percentile(gray, 99), border_frac=1 - mask.mean(),
        crop_area_frac=ch * cw / (h * w), crop_mean=cgray.mean(), crop_std=cgray.std(),
        n_components=int((areas >= 20).sum()),
        largest_vs_crop=largest_bbox / (ch * cw), corner_mean=corners.mean(),
        **{f'h{i}': v for i, v in enumerate(hist)}))
    raw_thumbs.append(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
    crop_thumbs.append(cv2.resize(cgray, (64, 64), interpolation=cv2.INTER_AREA))

with open(os.path.join(OUT, 'image_stats.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
np.savez_compressed(os.path.join(OUT, 'thumbs.npz'), raw=np.array(raw_thumbs), crop=np.array(crop_thumbs))
print(len(rows), "images")
