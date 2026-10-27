# Why were pool images left out of split.csv? Reasons per (source, label), plus
# a sheet of removed healthy images next to their closest Testing/external image.
import csv, os
from collections import Counter
import numpy as np, torch, cv2
from common import ROOT, pool_files, file_hash
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, 'image_stats.csv'))))
th = np.load(os.path.join(D, 'thumbs.npz'))['raw'].astype(np.float32) / 255
kept = {r['md5'] for r in csv.DictReader(open(os.path.join(D, '..', 'leak_free_split', 'split.csv')))}
T = 0.02
seen, exact = set(), Counter()
for p, y, s in pool_files():
    h = file_hash(p)
    if h in seen: exact[(s, y)] += 1
    seen.add(h)
print("(a) exact duplicates within pool (later copies dropped):", dict(sorted(exact.items())), "total", sum(exact.values()))
pool_i = {}
for i, r in enumerate(rows):
    if r['source'] in ('original', 'new'): pool_i.setdefault(r['md5'], i)
tst = [i for i, r in enumerate(rows) if r['source'] == 'testing']; ext = [i for i, r in enumerate(rows) if r['source'] == 'external']
md5_t = {rows[i]['md5'] for i in tst}; md5_e = {rows[i]['md5'] for i in ext}
pi = np.array(sorted(pool_i.values()))
flat = lambda ix, flip=False: torch.from_numpy((th[ix][:, :, ::-1] if flip else th[ix]).copy().reshape(len(ix), -1))
def nearest(ref):
    a, r, rm = flat(pi), flat(ref), flat(ref, True)
    d = torch.minimum(torch.cdist(a, r, p=1), torch.cdist(a, rm, p=1)) / a.shape[1]
    v, j = d.min(1); return v.numpy(), np.array(ref)[j.numpy()]
dt, jt = nearest(tst); de, je = nearest(ext)
reasons = Counter(); removed_h = []
for k, i in enumerate(pi):
    r = rows[i]; key = (r['source'], int(r['label']))
    if r['md5'] in kept: continue
    if r['md5'] in md5_t: why = 'b md5=Testing'
    elif r['md5'] in md5_e: why = 'c md5=external'
    elif dt[k] < T: why = 'b near Testing'
    elif de[k] < T: why = 'c near external'
    else: why = 'd same group as a held-out image (transitive)'
    reasons[(why, key)] += 1
    if key[1] == 0 and why.startswith(('b near', 'c near')):
        removed_h.append((k, i, why))
for why in sorted({w for w, _ in reasons}):
    print(f"{why:45s}", {f"{s}/{'tumor' if l else 'no'}": n for (w, (s, l)), n in sorted(reasons.items()) if w == why})
print("total removed:", sum(reasons.values()))
rng = np.random.default_rng(0)
pick = rng.choice(len(removed_h), 12, replace=False)
load = lambda i: cv2.resize(cv2.imread(rows[i]['path'] if os.path.isabs(rows[i]['path']) else os.path.join(ROOT, rows[i]['path'])), (128, 128))
tiles = []
for n in pick:
    k, i, why = removed_h[n]
    j, d = (jt[k], dt[k]) if why.startswith('b') else (je[k], de[k])
    t = np.hstack([load(i), load(j)]); cv2.putText(t, f"{d:.3f} {rows[i]['source'][0]}->{rows[j]['source'][:3]}", (2, 12), 0, 0.4, (0, 255, 255), 1)
    tiles.append(np.pad(t, ((2, 2), (2, 2), (0, 0))))
cv2.imwrite(os.path.join(D, 'removed_healthy_pairs.png'), np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:])]))
dd = np.array([min(dt[k], de[k]) for k, _, _ in removed_h])
print("removed healthy (direct near-dup) distance histogram:", np.histogram(dd, [0, .005, .01, .015, .02])[0].tolist(), "bins [0,.005,.01,.015,.02]")
