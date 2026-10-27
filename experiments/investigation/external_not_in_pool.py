# External-set baseline restricted to distinct images NOT byte-identical to any
# training-pool file. Uses eval_baseline.py's logged predictions; no model run.
import csv, os
from common import pool_files, file_hash
pool_md5 = {file_hash(p) for p, *_ in pool_files()}
rows = [r for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), 'baseline_predictions.csv'))) if r['set'] == 'external']
print(f"external distinct {len(rows)}, in pool {sum(r['md5'] in pool_md5 for r in rows)}")
rest = [r for r in rows if r['md5'] not in pool_md5]
for r in rest:
    print(f"{os.path.basename(r['path']):20s} label {r['label']}  p_combined {float(r['p_combined']):.3f}  p_original {float(r['p_original']):.3f}")
for m in ['p_combined', 'p_original']:
    print(m, "correct", sum((float(r[m]) > 0.5) == (r['label'] == '1') for r in rest), "of", len(rest))
