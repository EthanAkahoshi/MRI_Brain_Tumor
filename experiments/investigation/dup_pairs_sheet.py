# Contact sheet of random pool->pool nearest-neighbour pairs in distance bands
# (raw thumbnails), to eyeball where "same scan" stops.
import os, csv
import numpy as np, cv2
from common import ROOT, EXTERNAL
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, 'image_stats.csv'))))
z = np.load(os.path.join(D, 'nn.npz')); u = z['u']
d, arg, q = z['raw|pool->pool|d'], z['raw|pool->pool|arg'], z['raw|pool->pool|q']
rng = np.random.default_rng(0)
def load(i):
    p = rows[u[i]]['path']; im = cv2.imread(p if os.path.isabs(p) else os.path.join(ROOT, p))
    return cv2.resize(im, (128, 128))
bands = [(.008, .012), (.012, .016), (.016, .02), (.02, .025), (.025, .03), (.03, .04)]
lines = []
for lo, hi in bands:
    idx = np.where((d >= lo) & (d < hi))[0]; pick = rng.choice(idx, min(6, len(idx)), replace=False)
    tiles = []
    for k in pick:
        a, b = load(q[k]), load(arg[k])
        pair = np.hstack([a, b]); cv2.putText(pair, f"{d[k]:.3f} {rows[u[q[k]]]['source'][0]}{rows[u[arg[k]]]['source'][0]}", (2, 12), 0, 0.4, (0, 255, 255), 1)
        tiles.append(np.pad(pair, ((2, 2), (2, 2), (0, 0))))
    lines.append(np.hstack(tiles))
cv2.imwrite(os.path.join(D, 'dup_pairs_raw.png'), np.vstack(lines))
