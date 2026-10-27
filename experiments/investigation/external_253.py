# Re-scores the external set on all 253 files using the predictions already
# logged by eval_baseline.py (byte-identical files get the same prediction).
# No model is run here, so this is not a new external evaluation.
import csv, os
from collections import Counter
from common import external_files, file_hash
rows = {r['md5']: r for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), 'baseline_predictions.csv'))) if r['set'] == 'external'}
files = [(p, y, file_hash(p)) for p, y, _ in external_files()]
labels_by_md5 = {}
for p, y, h in files: labels_by_md5.setdefault(h, set()).add(y)
print("md5s with conflicting labels:", sum(len(v) > 1 for v in labels_by_md5.values()))
print("files per class:", Counter(y for _, y, _ in files), " duplicate files:", len(files) - len(labels_by_md5))
for m in ['p_combined', 'p_original']:
    tp = sum(1 for _, y, h in files if y == 1 and float(rows[h][m]) > 0.5); P = sum(y == 1 for _, y, _ in files)
    tn = sum(1 for _, y, h in files if y == 0 and float(rows[h][m]) <= 0.5); N = len(files) - P
    print(f"{m}: all 253 -> tumours caught {tp}/{P} ({tp/P:.4f}), healthy cleared {tn}/{N} ({tn/N:.4f}), acc {(tp+tn)/len(files):.4f}, bal {(tp/P+tn/N)/2:.4f}")
