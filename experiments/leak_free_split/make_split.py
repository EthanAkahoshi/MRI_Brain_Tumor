# Deterministic leak-free split of the training pool (datasets/no + datasets/yes), v2.
# 1. Drop exact duplicates (MD5, first copy wins, mainTrain.py order).
# 2. Near-duplicate distance between two images = the smallest mean-abs-diff of
#    their 64x64 grayscale thumbnails (INTER_AREA, 0-1 scale) over: raw image,
#    crop_to_brain image (what the model sees), each also compared mirrored.
# 3. Drop every pool image that shares an MD5 with, or is within THRESHOLD
#    of, any Testing or external image, so both stay genuinely held out.
#    Then link remaining pool pairs < THRESHOLD (single linkage) into groups.
# 4. StratifiedGroupKFold on class x source, seed 0: 1/5 of groups -> test,
#    then 1/5 of the rest -> val (~64/16/20).
# v1 (raw thumbnail only, threshold 0.02) is kept as split_v1.csv; the Reviewer
# rejected it because same-scan copies sat just above 0.02.
# Run from repo root: .venv\Scripts\python experiments\leak_free_split\make_split.py
import csv, os, sys
from collections import Counter
import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'investigation'))
from common import ROOT, EXTERNAL, pool_files, benchmark_files, external_files, image_files, file_hash, rel
from model import crop_to_brain

THRESHOLD = 0.045  # past the same-patient band (Reviewer's bands.png); see TEAM_NOTES.md
MAX_GROUP_FRACTION = 0.02  # fail loudly if single linkage chains into a giant group
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split.csv')
torch.set_num_threads(8)

held_md5 = {file_hash(p) for p in list(image_files(os.path.join(ROOT, 'datasets', 'Testing'))) + list(image_files(EXTERNAL))}
items, seen, labels_by_md5 = [], set(), {}
for p, y, s in pool_files():
    h = file_hash(p)
    labels_by_md5.setdefault(h, set()).add(y)
    if h not in seen:
        seen.add(h); items.append((p, y, s, h))
print("pool md5s with conflicting labels:", sum(len(v) > 1 for v in labels_by_md5.values()))
n_pool = len(items)
held = [(p, y, s, file_hash(p)) for p, y, s in benchmark_files() + external_files()]
everything = items + held


def thumb(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)


raw, crop = [], []
for p, *_ in everything:
    img = cv2.imread(p)
    raw.append(thumb(img)); crop.append(thumb(crop_to_brain(img)))
views = []
for t in (np.array(raw), np.array(crop)):
    t = t.astype(np.float32) / 255
    views.append((torch.from_numpy(t.reshape(len(t), -1)), torch.from_numpy(t[:, :, ::-1].copy().reshape(len(t), -1))))


def distance(rows_ix, cols_ix):
    # Smallest MAD over raw/crop views, each plain and mirrored
    d = None
    for a, m in views:
        r, c = a[rows_ix], a[cols_ix]
        v = torch.minimum(torch.cdist(r, c, p=1), torch.cdist(r, m[cols_ix], p=1)) / a.shape[1]
        d = v if d is None else torch.minimum(d, v)
    return d


# Edges of the near-duplicate graph, computed once
all_ix = torch.arange(len(everything))
edges = []
for s in range(0, len(everything), 512):
    d = distance(all_ix[s:s + 512], all_ix)
    i, j = (d < THRESHOLD).nonzero(as_tuple=True)
    edges.append(torch.stack([i + s, j], 1).numpy())
edges = np.concatenate(edges)
edges = edges[edges[:, 0] != edges[:, 1]]

# Held-out rule: drop any pool image that is byte-identical to, or directly
# within THRESHOLD of, a Testing/external image. (Grouping through held-out
# images chains Testing's 1,584 similar scans into one giant group.)
is_held = np.arange(len(everything)) >= n_pool
touch = np.zeros(len(everything), bool)
touch[edges[is_held[edges[:, 1]], 0]] = True
touch |= np.array([i < n_pool and items[i][3] in held_md5 for i in range(len(everything))])
touch &= ~is_held
keep = [i for i in range(n_pool) if not touch[i]]
removed = Counter((items[i][2], items[i][1]) for i in range(n_pool) if touch[i])
md5_removed = sum(items[i][3] in held_md5 for i in range(n_pool))
print(f"pool md5-distinct {n_pool}; removed {n_pool - len(keep)} within {THRESHOLD} of Testing/external "
      f"({md5_removed} byte-identical); by (source,label): {dict(sorted(removed.items()))}")

# Single linkage over the remaining pool only
kept_mask = np.zeros(len(everything), bool); kept_mask[keep] = True
parent = list(range(len(everything)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i, j in edges[kept_mask[edges[:, 0]] & kept_mask[edges[:, 1]]]:
    parent[find(int(i))] = find(int(j))
roots = np.array([find(i) for i in range(len(everything))])
largest = np.bincount(np.unique(roots[keep], return_inverse=True)[1]).max()
print(f"single-linkage over kept pool: largest group {largest} ({largest / len(keep):.2%} of kept pool)")
assert largest <= MAX_GROUP_FRACTION * len(keep), "chaining blew up a group; switch linkage"

y = np.array([items[i][1] for i in keep]); src = np.array([items[i][2] for i in keep])
key = y * 2 + (src == 'new')
_, groups = np.unique(roots[keep], return_inverse=True)
idx = np.arange(len(keep))
def group_split(ix):
    tr, ho = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(ix, key[ix], groups[ix]))
    return ix[tr], ix[ho]
tr, te = group_split(idx); tr, va = group_split(tr)
split = np.empty(len(keep), dtype=object); split[tr], split[va], split[te] = 'train', 'val', 'test'

for g in np.unique(groups):
    assert len(set(split[groups == g])) == 1, g
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['path', 'md5', 'source', 'label', 'group', 'split'])
    for k, i in enumerate(keep):
        p, lab, s, h = items[i]
        w.writerow([rel(p), h, s, lab, int(groups[k]), split[k]])
gs = np.bincount(groups)
print(f"kept {len(keep)} images in {len(gs)} groups; largest {gs.max()}; group-size distribution {dict(sorted(Counter(gs.tolist()).items()))}")
for sp in ['train', 'val', 'test']:
    c = Counter(zip(src[split == sp], y[split == sp]))
    print(f"{sp:5s} n={(split == sp).sum():5d}  healthy={(y[split == sp] == 0).sum():4d}  " +
          "  ".join(f"{s}/{'tumor' if l else 'no'}={c[(s, l)]}" for s in ['original', 'new'] for l in [0, 1]))

# Closest remaining cross-split pairs (for the Reviewer to eyeball)
kix = torch.tensor(keep)
best = []
for s in range(0, len(keep), 512):
    d = distance(kix[s:s + 512], kix)
    cross = torch.from_numpy(split[s:s + 512, None] != split[None, :])
    d[~cross] = 9
    v, j = d.min(1)
    best += [(float(v[a]), s + a, int(j[a])) for a in range(len(v))]
seen_pairs = set()
print("10 closest cross-split pairs:")
for dist, a, b in sorted(best):
    if (min(a, b), max(a, b)) in seen_pairs:
        continue
    seen_pairs.add((min(a, b), max(a, b)))
    print(f"  {dist:.4f}  {rel(items[keep[a]][0])} ({split[a]})  <->  {rel(items[keep[b]][0])} ({split[b]})")
    if len(seen_pairs) == 10:
        break
