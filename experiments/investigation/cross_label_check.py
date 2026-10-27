# Near-duplicate pairs (<0.02, raw 64px gray, flip) and byte-identical files
# across all sets: external<->Testing overlap and any pair with conflicting labels.
import csv, os
from collections import Counter
import numpy as np, torch
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, 'image_stats.csv'))))
th = np.load(os.path.join(D, 'thumbs.npz'))['raw'].astype(np.float32) / 255
lab = {}
for r in rows: lab.setdefault(r['md5'], set()).add(r['label'])
print("md5s with conflicting labels across all sets:", sum(len(v) > 1 for v in lab.values()))
grp = lambda s: 'pool' if s in ('original', 'new') else s
a = torch.from_numpy(th.reshape(len(th), -1)); m = torch.from_numpy(th[:, :, ::-1].copy().reshape(len(th), -1))
pairs, conflicts = Counter(), Counter(); ex = []
for s in range(0, len(a), 512):
    d = torch.minimum(torch.cdist(a[s:s+512], a, p=1), torch.cdist(a[s:s+512], m, p=1)) / a.shape[1]
    for i, j in (d < 0.02).nonzero().tolist():
        i += s
        if i >= j or rows[i]['md5'] == rows[j]['md5']: continue
        k = tuple(sorted([grp(rows[i]['source']), grp(rows[j]['source'])]))
        pairs[k] += 1
        if rows[i]['label'] != rows[j]['label']:
            conflicts[k] += 1
            if len(ex) < 8: ex.append((rows[i]['path'], rows[j]['path'], round(float(d[i - s, j]), 4)))
ext_md5 = {r['md5'] for r in rows if r['source'] == 'external'}; tst_md5 = {r['md5'] for r in rows if r['source'] == 'testing'}
print("external md5 also in Testing:", len(ext_md5 & tst_md5))
print("near-dup (non-identical) pairs by set pair:", dict(pairs))
print("  of which conflicting labels:", dict(conflicts))
for e in ex: print("   ", e)
