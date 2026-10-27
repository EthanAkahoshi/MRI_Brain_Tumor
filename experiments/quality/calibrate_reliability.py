# Distribution of reliability signals on the leak-free VALIDATION split (not test,
# Testing or external), to set RELIABILITY_THRESHOLDS: background heat, and the
# mirror test's score change and heatmap agreement, split by true class.
import csv, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradcam import grad_cam  # noqa: E402
from model import DECISION_THRESHOLD, load_model, preprocess, tumor_probability  # noqa: E402
from reliability import background_heat, heatmap_agreement  # noqa: E402

model = load_model(); layer = model.features[3][2]
rows = [r for r in csv.DictReader(open('experiments/leak_free_split/split.csv')) if r['split'] == 'val']
out = {'heat': [], 'dscore': [], 'agree': [], 'label': []}
for r in rows:
    img = cv2.imread(r['path']); t = preprocess(img); p = tumor_probability(model, t); lab = int(p >= DECISION_THRESHOLD)
    cam = grad_cam(model, t, lab, layer)
    t2 = preprocess(cv2.flip(img, 1)); p2 = tumor_probability(model, t2); cam2 = grad_cam(model, t2, lab, layer)[:, ::-1]
    out['heat'].append(background_heat(cam, t)); out['dscore'].append(abs(p2 - p))
    out['agree'].append(heatmap_agreement(cam, cam2)); out['label'].append(lab)
res = {'n_val': len(rows)}
lab = np.array(out['label'])
for k in ('heat', 'dscore', 'agree'):
    v = np.array(out[k])
    res[k] = {cls: {q: round(float(np.percentile(v[lab == c], q)), 3) for q in (5, 50, 95)}
              for cls, c in (('pred_no_tumor', 0), ('pred_tumor', 1))}
print(json.dumps(res, indent=1))
