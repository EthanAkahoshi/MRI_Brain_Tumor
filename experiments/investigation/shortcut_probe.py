# Can non-anatomical image statistics predict source or label? Logistic
# regression, 5-fold StratifiedGroupKFold over the leak-free split.csv pool
# (groups = near-duplicate groups), seed 0.
import csv, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
D = os.path.dirname(os.path.abspath(__file__))
stats = {}
for r in csv.DictReader(open(os.path.join(D, 'image_stats.csv'))): stats.setdefault(r['md5'], r)
split = list(csv.DictReader(open(os.path.join(D, '..', 'leak_free_split', 'split.csv'))))
META = ['height', 'width', 'aspect', 'bpp', 'is_png', 'chan_diff']
INTEN = ['mean', 'std', 'p99', 'border_frac', 'crop_area_frac', 'crop_mean', 'crop_std', 'n_components',
         'largest_vs_crop', 'corner_mean'] + [f'h{i}' for i in range(16)]
def feats(r, cols):
    r = dict(r); r['bpp'] = float(r['bytes']) / (float(r['height']) * float(r['width'])); r['is_png'] = float(r['ext'] == '.png')
    return [float(r[c]) for c in cols]
rows = [stats[s['md5']] for s in split]
y = np.array([int(s['label']) for s in split]); src = np.array([s['source'] == 'new' for s in split]).astype(int)
g = np.array([int(s['group']) for s in split])
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

print("Per source x class medians (pool, leak-free split.csv):")
cols = ['height', 'width', 'aspect', 'chan_diff', 'mean', 'std', 'border_frac', 'crop_area_frac', 'crop_mean', 'largest_vs_crop', 'corner_mean']
groups_all = {'original/no': (src == 0) & (y == 0), 'original/tumor': (src == 0) & (y == 1), 'new/no': (src == 1) & (y == 0), 'new/tumor': (src == 1) & (y == 1)}
held = {k: [r for r in stats.values() if r['source'] == k] for k in ['testing', 'external']}
print(f"{'':15s}" + "".join(f"{c[:12]:>13s}" for c in cols) + "   formats")
for name, mk in groups_all.items():
    rr = [rows[i] for i in np.where(mk)[0]]
    print(f"{name:15s}" + "".join(f"{np.median([float(r[c]) for r in rr]):13.3f}" for c in cols), "  ", dict(sorted(__import__('collections').Counter(r['ext'] for r in rr).items())))
for k, rr in held.items():
    print(f"{k:15s}" + "".join(f"{np.median([float(r[c]) for r in rr]):13.3f}" for c in cols), "  ", dict(sorted(__import__('collections').Counter(r['ext'] for r in rr).items())))
print("fraction truly grayscale (chan_diff < 1):", {n: round(float(np.mean([float(rows[i]['chan_diff']) < 1 for i in np.where(mk)[0]])), 3) for n, mk in groups_all.items()},
      {k: round(float(np.mean([float(r['chan_diff']) < 1 for r in rr])), 3) for k, rr in held.items()})

print("\n5-fold grouped CV, logistic regression (accuracy / balanced accuracy):")
for fname, cols_ in [('meta', META), ('intensity', INTEN), ('meta+intensity', META + INTEN)]:
    X = np.array([feats(r, cols_) for r in rows])
    for tname, t, mask in [('source', src, np.ones(len(y), bool)), ('label (all)', y, np.ones(len(y), bool)),
                           ('label within original', y, src == 0), ('label within new', y, src == 1)]:
        p = cross_val_predict(clf(), X[mask], t[mask], groups=g[mask], cv=cv)
        print(f"  {fname:15s} -> {tname:22s} acc {accuracy_score(t[mask], p):.4f}  bal {balanced_accuracy_score(t[mask], p):.4f}  (n={mask.sum()})")
    if fname == 'meta+intensity':
        m = clf().fit(X, src)
        for k, rr in held.items():
            pn = m.predict(np.array([feats(r, cols_) for r in rr]))
            print(f"  source classifier on {k}: {pn.mean():.3f} predicted 'new Kaggle' (n={len(rr)})")
