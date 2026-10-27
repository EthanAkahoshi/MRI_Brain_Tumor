"""Healthy-only external test on IXI T1 slices (new hospitals, never used for training).

All IXI subjects are healthy volunteers, so the only measurable error is a false
tumour call: this reports specificity per slice and per subject, plus what the
quality and reliability checks say about this unfamiliar source.
Writes docs/ixi_evaluation.json.
"""
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import reliability  # noqa: E402
from model import load_model  # noqa: E402

model = load_model()
layer = model.features[3][2]
rows = list(csv.DictReader(open('datasets/IXI/slices.csv')))

flags = []
by_subject = collections.defaultdict(list)
statuses = collections.Counter()
flagged_review = []
warnings = collections.Counter()
for r in rows:
    out = reliability.run(cv2.imread(r['path']), model, layer)
    flagged = out['label'] == 1
    flags.append(flagged)
    by_subject[r['subject']].append(flagged)
    statuses[out['reliability']['status']] += 1
    flagged_review.append(out['reliability']['human_review'])
    for check in out['quality']['checks']:
        warnings[check['key']] += check['status'] == 'warning'

flags = np.array(flags)
subject_any = np.array([any(v) for v in by_subject.values()])
result = {
    'n_slices': int(len(flags)),
    'n_subjects': len(by_subject),
    'slice_specificity': float(1 - flags.mean()),
    'false_tumour_slices': int(flags.sum()),
    'subjects_with_any_false_tumour': int(subject_any.sum()),
    'reliability_status_counts': dict(statuses),
    # Does the reliability layer catch the model's mistakes? (share of false tumour calls sent to review)
    'false_tumour_sent_to_review': int((flags & np.array(flagged_review)).sum()),
    'correct_slices_sent_to_review': int((~flags & np.array(flagged_review)).sum()),
    'quality_warning_counts': dict(warnings),
}
os.makedirs('docs', exist_ok=True)
json.dump(result, open('docs/ixi_evaluation.json', 'w'), indent=1)
print(json.dumps(result, indent=1))
