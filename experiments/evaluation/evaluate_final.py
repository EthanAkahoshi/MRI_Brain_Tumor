"""Final evaluation of BrainTumorFinal.pt: metrics on the leak-free test split, the
Testing benchmark (run once) and the external set (its last allowed evaluation),
plus latency and memory. Writes docs/evaluation.json and docs/evaluation.md."""
import csv, json, os, statistics, sys, time
import cv2, numpy as np, psutil, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score  # noqa: E402
from model import DECISION_THRESHOLD, MODEL_PATH, load_model, preprocess, tumor_probability  # noqa: E402
import reliability  # noqa: E402

proc = psutil.Process()
rss_start = proc.memory_info().rss
model = load_model(); layer = model.features[3][2]
rss_model = proc.memory_info().rss


def files(folder, label_of):
    out = []
    for r, _, fs in os.walk(folder):
        for n in sorted(fs):
            if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                out.append((os.path.join(r, n), label_of(r)))
    return sorted(out)


def metrics(probs, labels):
    pred = (probs >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {'n': int(len(labels)), 'n_tumor': int(tp + fn), 'n_healthy': int(tn + fp),
            'accuracy': float((tp + tn) / len(labels)), 'balanced_accuracy': float((sens + spec) / 2),
            'precision': float(precision_score(labels, pred, zero_division=0)), 'recall_sensitivity': float(sens),
            'specificity': float(spec), 'f1': float(f1_score(labels, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(labels, probs)),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}


def score(items):
    probs = np.array([tumor_probability(model, preprocess(cv2.imread(p))) for p, _ in items])
    return metrics(probs, np.array([l for _, l in items]))


split = [r for r in csv.DictReader(open('experiments/leak_free_split/split.csv')) if r['split'] == 'test']
sets = {
    'leak_free_test': [(r['path'], int(r['label'])) for r in split],
    'testing_benchmark': files('datasets/Testing', lambda r: 0 if os.path.basename(r) == 'notumor' else 1),
    'external_253': files(r'Z:\GoogleDownloads\Brain\pred\brain_tumor_dataset', lambda r: 1 if os.path.basename(r) == 'yes' else 0),
}
results = {name: score(items) for name, items in sets.items()}
for s in ('original', 'new'):
    results[f'leak_free_test_{s}_source'] = score([(r['path'], int(r['label'])) for r in split if r['source'] == s])

# Latency on 60 test images: model only, preprocess + model, and the full web pipeline
imgs = [cv2.imread(r['path']) for r in split[:60]]
torch.set_num_threads(torch.get_num_threads())
def timed(fn):
    fn(imgs[0])  # warm-up
    t = []
    for im in imgs:
        s = time.perf_counter(); fn(im); t.append((time.perf_counter() - s) * 1000)
    return {'mean_ms': round(statistics.mean(t), 2), 'p95_ms': round(float(np.percentile(t, 95)), 2)}
tensors = {id(im): preprocess(im) for im in imgs}
results['latency_cpu'] = {
    'threads': torch.get_num_threads(),
    'model_only': timed(lambda im: tumor_probability(model, tensors[id(im)])),
    'preprocess_plus_model': timed(lambda im: tumor_probability(model, preprocess(im))),
    'full_pipeline_quality_gradcam_reliability': timed(lambda im: reliability.run(im, model, layer)),
}
results['memory'] = {
    'weights_file_mb': round(os.path.getsize(MODEL_PATH) / 2**20, 2),
    'parameters': int(sum(p.numel() for p in model.parameters())),
    'process_rss_mb_after_load': round(rss_model / 2**20, 1),
    'model_load_rss_increase_mb': round((rss_model - rss_start) / 2**20, 1),
    'process_peak_rss_mb': round(proc.memory_info().peak_wset / 2**20, 1) if hasattr(proc.memory_info(), 'peak_wset') else None,
}
os.makedirs('docs', exist_ok=True)
json.dump(results, open('docs/evaluation.json', 'w'), indent=1)
lines = ['# Final model evaluation (`BrainTumorFinal.pt`, threshold 0.5)', '',
         '| Set | n (tumour / healthy) | Acc | Bal. acc | Precision | Recall | Specificity | F1 | ROC-AUC | TN / FP / FN / TP |',
         '|---|---|---|---|---|---|---|---|---|---|']
for k, r in results.items():
    if 'confusion_matrix' in r:
        c = r['confusion_matrix']
        lines.append(f"| {k} | {r['n']} ({r['n_tumor']} / {r['n_healthy']}) | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['precision']:.3f} | {r['recall_sensitivity']:.3f} | {r['specificity']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {c['tn']} / {c['fp']} / {c['fn']} / {c['tp']} |")
L = results['latency_cpu']; M = results['memory']
lines += ['', f"Latency (CPU, {L['threads']} threads, 60 images): model {L['model_only']['mean_ms']} ms (p95 {L['model_only']['p95_ms']}), "
          f"preprocess+model {L['preprocess_plus_model']['mean_ms']} ms, full web pipeline {L['full_pipeline_quality_gradcam_reliability']['mean_ms']} ms (p95 {L['full_pipeline_quality_gradcam_reliability']['p95_ms']}).",
          f"Memory: weights {M['weights_file_mb']} MB, {M['parameters']:,} parameters, process RSS after load {M['process_rss_mb_after_load']} MB.",
          '', 'external_253 is held out from the training pool (near-duplicates removed) but comes from the ORIGINAL source, not an independent hospital.']
open('docs/evaluation.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
