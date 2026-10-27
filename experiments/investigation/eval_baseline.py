# Baseline: evaluate the two existing models on (a) mainTrain.py's random test
# split (reproduced without training), (b) datasets/Testing, (c) the external set.
# This is External-set evaluation #1 of 3. Do not re-run casually.
# Run from repo root: .venv\Scripts\python experiments\investigation\eval_baseline.py
import csv
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import ROOT, pool_files, benchmark_files, external_files, file_hash, rel
from model import BrainTumorCNN, preprocess

torch.manual_seed(0)
MODELS = {'combined': 'BrainTumorBest.pt', 'original': 'BrainTumorBest_original-data.pt'}
models = {}
for name, f in MODELS.items():
    m = BrainTumorCNN()
    m.load_state_dict(torch.load(os.path.join(ROOT, f), weights_only=True))
    models[name] = m.eval()


def dedupe(files, skip):
    # Same rule as mainTrain.load_images: keep the first copy of each MD5
    seen, kept = set(skip), []
    for p, y, s in files:
        h = file_hash(p)
        if h in seen:
            continue
        seen.add(h)
        kept.append((p, y, s, h))
    return kept


def predict(items):
    probs = {n: [] for n in models}
    for start in range(0, len(items), 64):
        batch = torch.cat([preprocess(cv2.imread(p)) for p, *_ in items[start:start + 64]])
        with torch.no_grad():
            for n, m in models.items():
                probs[n].append(torch.softmax(m(batch), 1)[:, 1].numpy())
    return {n: np.concatenate(v) for n, v in probs.items()}


def split_test_mask(labels):
    # Exactly mainTrain.py's two train_test_split calls, on indices
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=labels)
    tr, va = train_test_split(tr, test_size=0.2, random_state=0, stratify=labels[tr])
    split = np.empty(len(labels), dtype=object)
    split[tr], split[va], split[te] = 'train', 'val', 'test'
    return split


def metrics(y, p):
    pred = (p > 0.5).astype(int)  # == argmax for 2 classes
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else float('nan')
    return (f"n={len(y):5d} acc {(tp + tn) / len(y):.4f}  bal {(sens + spec) / 2:.4f}  "
            f"sens {sens:.4f} ({tp}/{tp + fn})  spec {spec:.4f} ({tn}/{tn + fp})  AUC {auc:.4f}")


bench = benchmark_files()
bench_hashes = {file_hash(p) for p, *_ in bench}
pool = dedupe(pool_files(), bench_hashes)
orig_pool = dedupe([f for f in pool_files() if f[2] == 'original'], bench_hashes)
bench_u = dedupe(bench, set())
ext = dedupe(external_files(), set())
print(f"pool kept {len(pool)} (original {sum(s == 'original' for _, _, s, _ in pool)}, "
      f"new {sum(s == 'new' for _, _, s, _ in pool)}); original-only pool {len(orig_pool)}; "
      f"Testing unique {len(bench_u)} of {len(bench)}; external unique {len(ext)} of {len(external_files())}")

sets = {}
for name, items in [('pool', pool), ('orig_pool', orig_pool), ('testing', bench_u), ('external', ext)]:
    y = np.array([it[1] for it in items])
    sets[name] = (items, y, predict(items))

y = sets['pool'][1]; pool_split = split_test_mask(y)
oy = sets['orig_pool'][1]; orig_split = split_test_mask(oy)

rows = []
for name, split in [('pool', pool_split), ('orig_pool', orig_split)]:
    items, yy, pr = sets[name]
    for i, it in enumerate(items):
        rows.append([name, rel(it[0]), it[3], it[2], it[1], split[i], pr['combined'][i], pr['original'][i]])
for name in ['testing', 'external']:
    items, yy, pr = sets[name]
    for i, it in enumerate(items):
        rows.append([name, rel(it[0]) if name != 'external' else it[0], it[3], it[2], it[1], '', pr['combined'][i], pr['original'][i]])
with open(os.path.join(os.path.dirname(__file__), 'baseline_predictions.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['set', 'path', 'md5', 'source', 'label', 'split', 'p_combined', 'p_original']); w.writerows(rows)

for mname in models:
    print(f"\n=== model {MODELS[mname]} ===")
    items, yy, pr = sets['pool']; t = pool_split == 'test'
    print("combined random test split ", metrics(yy[t], pr[mname][t]))
    src = np.array([it[2] for it in items])
    for s in ['original', 'new']:
        k = t & (src == s)
        print(f"   source={s:8s}            ", metrics(yy[k], pr[mname][k]))
    items, yy, pr = sets['orig_pool']; t = orig_split == 'test'
    print("original-only random test   ", metrics(yy[t], pr[mname][t]))
    for name in ['testing', 'external']:
        items, yy, pr = sets[name]
        print(f"{name:28s}", metrics(yy, pr[mname]))

# Leakage check: how many test-split images were in the other model's training split?
items, _, _ = sets['pool']; oitems, _, _ = sets['orig_pool']
orig_train = {it[3] for it, s in zip(oitems, orig_split) if s != 'test'}
comb_test = {it[3] for it, s in zip(items, pool_split) if s == 'test'}
print(f"\ncombined-split test images that are in the original-only train+val split: "
      f"{len(comb_test & orig_train)} of {len(comb_test)}")
