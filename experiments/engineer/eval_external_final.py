# External eval #3 (final): both final v2 seeds, thresholds fixed on validation.
import hashlib, json, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from train import metrics, predict  # noqa: E402
from model_bn import BrainTumorBNCNN  # noqa: E402
from model import preprocess_legacy as preprocess  # noqa: E402  the final v2-split models used the legacy pipeline

EXT = r'Z:\GoogleDownloads\Brain\pred\brain_tumor_dataset'
paths, labels, seen, keep = [], [], set(), []
for sub, label in (('no', 0), ('yes', 1)):
    for name in sorted(os.listdir(os.path.join(EXT, sub))):
        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(EXT, sub, name)
            d = hashlib.md5(open(p, 'rb').read()).hexdigest()
            keep.append(d not in seen); seen.add(d)
            paths.append(p); labels.append(label)
labels, keep = np.array(labels), np.array(keep)
# predict() expects uint8 0-255 like the training cache (it divides by 255 itself)
images = np.round(np.concatenate([preprocess(cv2.imread(p)).numpy() for p in paths]) * 255).astype(np.uint8)
for seed in (0, 1):
    m = BrainTumorBNCNN(); m.load_state_dict(torch.load(f'models/final_srcbal_seed{seed}_v2.pt', weights_only=True)); m.eval()
    thr = json.load(open(f'models/final_srcbal_seed{seed}_v2.json'))['threshold']
    probs = predict(m, images)
    for name, mask in (('all 253 files', np.ones(len(labels), bool)), ('228 distinct', keep)):
        r = metrics(probs[mask], labels[mask], thr)
        print(f"seed{seed} thr {thr:.4f} {name}: " + ", ".join(f"{k} {r[k]:.4f}" if isinstance(r[k], float) else f"{k} {r[k]}" for k in ('n', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'tp', 'fn', 'tn', 'fp')))
