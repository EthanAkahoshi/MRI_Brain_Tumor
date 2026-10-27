# Distribution of quality measurements on the leak-free TRAINING split only
# (val/test/Testing/external never used), to set QUALITY_THRESHOLDS.
import csv, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from quality_analysis import measure  # noqa: E402

rows = [r for r in csv.DictReader(open('experiments/leak_free_split/split.csv')) if r['split'] == 'train']
ms = [measure(cv2.imread(r['path'])) for r in rows]
keys = ['sharpness', 'brightness', 'clipped_fraction', 'contrast', 'crop_coverage',
        'background_fraction', 'stray_regions', 'colour_spread']
out = {}
for k in keys:
    v = np.array([m[k] for m in ms], float)
    out[k] = {q: round(float(np.percentile(v, q)), 4) for q in (1, 5, 50, 95, 99)}
side = np.array([min(m['crop_width'] / m['width'], m['crop_height'] / m['height']) for m in ms])
out['crop_side_fraction'] = {q: round(float(np.percentile(side, q)), 4) for q in (1, 5, 50)}
short = np.array([min(m['width'], m['height']) for m in ms])
out['short_side'] = {q: int(np.percentile(short, q)) for q in (1, 5, 50)}
print(json.dumps({'n_train': len(rows), 'percentiles': out}, indent=1))
