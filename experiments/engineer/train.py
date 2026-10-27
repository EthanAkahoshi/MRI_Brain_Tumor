# Engineer training script on the leak-free split.
# Run from the repo root:
#   .venv/Scripts/python.exe experiments/engineer/train.py --seed 0 --balance classweight --aug base --out models/run1_bn_seed0.pt
# Final, once, for the chosen model only:
#   .venv/Scripts/python.exe experiments/engineer/train.py --eval-testing models/<chosen>.pt
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as shared_model  # noqa: E402  --size sets shared_model.INPUT_SIZE, which preprocess() reads
from model import preprocess_legacy as preprocess  # noqa: E402  "shared" = legacy pipeline (cache names unchanged)
from model_bn import BrainTumorBNCNN  # noqa: E402
from preprocess_v2 import preprocess_v2  # noqa: E402  run 4 only

PREP = {'shared': preprocess, 'v2': preprocess_v2}

CACHE_DIR = os.path.join(ROOT, 'experiments', 'engineer', 'cache')
TESTING_DIR = os.path.join(ROOT, 'datasets', 'Testing')
CL = torch.channels_last  # ~25% faster convs on CPU


def load_cached(paths, key, prep='shared', size=128):
    # preprocess() output is uint8/255, so storing it as uint8 is lossless
    os.makedirs(CACHE_DIR, exist_ok=True)
    shared_model.INPUT_SIZE = size
    cache = os.path.join(CACHE_DIR, key + ('' if prep == 'shared' else '_' + prep)
                         + ('' if size == 128 else f'_s{size}') + '.npy')
    if os.path.exists(cache):
        return np.load(cache)

    def one(p):
        image = cv2.imread(p if os.path.isabs(p) else os.path.join(ROOT, p))
        if image is None:
            raise RuntimeError(f"Could not load {p}")
        return np.round(PREP[prep](image)[0].numpy() * 255).astype(np.uint8)

    with ThreadPool(16) as pool:
        images = np.stack(pool.map(one, paths))
    np.save(cache, images)
    return images


def metrics(probs, labels, threshold, sources=None):
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    out = {'n': len(labels), 'n_tumor': tp + fn, 'n_healthy': tn + fp,
           'accuracy': (tp + tn) / len(labels), 'sensitivity': sens, 'specificity': spec,
           'balanced_accuracy': (sens + spec) / 2 if sens is not None and spec is not None else None,
           'roc_auc': roc_auc_score(labels, probs) if len(set(labels.tolist())) == 2 else None,
           'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}
    if sources is not None:
        out['per_source'] = {s: metrics(probs[sources == s], labels[sources == s], threshold)
                             for s in sorted(set(sources.tolist()))}
    return out


def best_threshold(probs, labels):
    # Maximise balanced accuracy on val; ties -> threshold closest to 0.5
    best = (-1, 0, 0.5)
    for t in np.unique(np.concatenate([probs, [0.5]])):
        b = metrics(probs, labels, t)['balanced_accuracy']
        key = (round(b, 10), -abs(t - 0.5), float(t))
        if key[:2] > best[:2]:
            best = key
    return best[2]


def predict(model, images, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).float().div_(255).contiguous(memory_format=CL)
            out.append(torch.softmax(model(x), 1)[:, 1].numpy())
    return np.concatenate(out)


def random_gamma(x):
    return x.clamp(1e-6, 1) ** float(np.exp(np.random.uniform(np.log(0.5), np.log(2.0))))


def random_noise(x):
    return (x + torch.randn_like(x) * np.random.uniform(0, 0.05)).clamp(0, 1)


def random_down_up(x):
    # Kills the resolution/sharpness cue (new-Kaggle 512px vs original ~240px)
    size = x.shape[-1]
    small = max(8, int(size * np.random.uniform(0.4, 1.0)))
    x = torch.nn.functional.interpolate(x[None], size=small, mode='area')
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)[0].clamp(0, 1)


def source_mean_bal(m):
    # Balanced accuracy averaged over sources (Investigator's suggested metric; logged, not used to select)
    return float(np.mean([v['balanced_accuracy'] for v in m['per_source'].values()]))


def make_transform(mode):
    geometric = [v2.RandomHorizontalFlip(),
                 v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))]
    if mode == 'base':  # same as mainTrain.py
        return v2.Compose(geometric + [v2.ColorJitter(brightness=0.2, contrast=0.2)])
    # strong: intensity/contrast changes so per-source intensity cues stop working
    return v2.Compose(geometric + [
        v2.Lambda(random_gamma),
        v2.ColorJitter(brightness=0.4, contrast=0.4),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        v2.RandomApply([v2.Lambda(random_noise)], p=0.5),
        v2.RandomApply([v2.Lambda(random_down_up)], p=0.5),
    ])


class Images(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, torch.from_numpy(labels).long(), transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(torch.from_numpy(self.images[i]).float().div_(255)), self.labels[i]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def eval_testing(weights):
    # The Testing benchmark, run once for the final chosen model, with the val threshold
    result_path = os.path.splitext(weights)[0] + '.json'
    with open(result_path) as f:
        result = json.load(f)
    paths = sorted(os.path.relpath(os.path.join(r, n), ROOT) for r, _, fs in os.walk(TESTING_DIR)
                   for n in fs if n.lower().endswith(('.jpg', '.jpeg', '.png')))
    labels = np.array([0 if os.path.basename(os.path.dirname(p)) == 'notumor' else 1 for p in paths])
    folders = np.array([os.path.basename(os.path.dirname(p)) for p in paths])
    # Same preprocessing and size the model was trained with
    run_args = result['args']
    images = load_cached(paths, 'testing_' + hashlib.md5('\n'.join(paths).encode()).hexdigest()[:12],
                         run_args.get('prep', 'shared'), run_args.get('size', 128))
    model = BrainTumorBNCNN().to(memory_format=CL)
    model.load_state_dict(torch.load(weights, weights_only=True))
    probs = predict(model, images)
    result['testing'] = metrics(probs, labels, result['threshold'], folders)
    result['testing_at_0.5'] = metrics(probs, labels, 0.5)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ('threshold', 'testing', 'testing_at_0.5')}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default=os.path.join(ROOT, 'experiments', 'leak_free_split', 'split.csv'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--balance', choices=['classweight', 'source'], default='classweight')
    ap.add_argument('--aug', choices=['base', 'strong'], default='base')
    ap.add_argument('--out', help='new .pt path; refuses to overwrite')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--max-minutes', type=float, default=25)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--prep', choices=list(PREP), default='shared', help='v2 = run 4 experimental preprocessing')
    ap.add_argument('--size', type=int, default=128, help='input size; shared preprocessing only')
    ap.add_argument('--eval-testing', metavar='WEIGHTS')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.eval_testing:
        return eval_testing(args.eval_testing)
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; never overwrite weights")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    seed_all(args.seed)
    with open(args.split, 'rb') as f:
        key = hashlib.md5(f.read()).hexdigest()[:12]
    with open(args.split, newline='') as f:
        rows = list(csv.DictReader(f))
    start = time.time()
    if args.size != 128 and args.prep != 'shared':
        sys.exit("--size only works with --prep shared (preprocess_v2 has its size fixed at import)")
    images = load_cached([r['path'] for r in rows], 'split_' + key, args.prep, args.size)
    labels = np.array([int(r['label']) for r in rows])
    sources = np.array([r['source'] for r in rows])
    split = np.array([r['split'] for r in rows])
    idx = {s: np.where(split == s)[0] for s in ('train', 'val', 'test')}
    print(f"split {key}: loaded {len(rows)} images in {time.time() - start:.0f}s", flush=True)
    for s in idx:
        print(s, dict(Counter(zip(sources[idx[s]], labels[idx[s]]))), flush=True)

    tr = idx['train']
    y_train = labels[tr]
    g = torch.Generator().manual_seed(args.seed)
    dataset = Images(images[tr], y_train, make_transform(args.aug))
    if args.balance == 'source':
        # Each source x class cell gets equal total weight per epoch; no loss weights
        cells = list(zip(sources[tr], y_train))
        counts = Counter(cells)
        weights = torch.tensor([1.0 / counts[c] for c in cells], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(tr), replacement=True, generator=g)
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler)
        loss_fn = nn.CrossEntropyLoss()
    else:
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=g)
        counts = np.bincount(y_train, minlength=2)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32))

    model = BrainTumorBNCNN().to(memory_format=CL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x.contiguous(memory_format=CL))
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y); total += len(y); correct += (out.argmax(1) == y).sum().item()
        val = metrics(predict(model, images[idx['val']]), labels[idx['val']], 0.5, sources[idx['val']])
        scheduler.step(val['balanced_accuracy'])
        improved = val['balanced_accuracy'] > best
        if improved:
            best, best_epoch = val['balanced_accuracy'], epoch
            torch.save(model.state_dict(), args.out)
        history.append({'epoch': epoch, 'train_loss': loss_sum / total, 'train_acc': correct / total,
                        'val_bal': val['balanced_accuracy'], 'val_sens': val['sensitivity'],
                        'val_spec': val['specificity'], 'val_src_bal': source_mean_bal(val), 'lr': optimizer.param_groups[0]['lr']})
        print(f"epoch {epoch} loss {loss_sum / total:.4f} train_acc {correct / total:.4f} "
              f"val_bal {val['balanced_accuracy']:.4f} sens {val['sensitivity']:.4f} spec {val['specificity']:.4f} src_bal {source_mean_bal(val):.4f} "
              f"lr {optimizer.param_groups[0]['lr']:.1e} {time.time() - t0:.0f}s{' *' if improved else ''}", flush=True)
        if epoch - best_epoch >= args.patience:
            print("early stop"); break
        if (time.time() - start) / 60 > args.max_minutes:
            print("time cap reached"); break

    model.load_state_dict(torch.load(args.out, weights_only=True))
    val_probs = predict(model, images[idx['val']])
    threshold = best_threshold(val_probs, labels[idx['val']])
    print(f"best epoch {best_epoch}, val threshold {threshold:.4f} (chosen on val before test)", flush=True)
    test_probs = predict(model, images[idx['test']])
    result = {'args': vars(args), 'split_md5': key, 'best_epoch': best_epoch, 'epochs_run': epoch,
              'wall_minutes': (time.time() - start) / 60, 'threshold': float(threshold),
              'val_at_0.5': metrics(val_probs, labels[idx['val']], 0.5, sources[idx['val']]),
              'val': metrics(val_probs, labels[idx['val']], threshold, sources[idx['val']]),
              'test': metrics(test_probs, labels[idx['test']], threshold, sources[idx['test']]),
              'test_at_0.5': metrics(test_probs, labels[idx['test']], 0.5),
              'history': history}
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(result, f, indent=1)
    for k in ('val', 'test'):
        m = result[k]
        print(f"{k} @thr: acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.4f} sens {m['sensitivity']:.4f} "
              f"spec {m['specificity']:.4f} auc {m['roc_auc']:.4f} src_bal {source_mean_bal(m):.4f} n={m['n']}")
        for s, ms in m['per_source'].items():
            print(f"   {s}: n={ms['n']} (tumor {ms['n_tumor']}, healthy {ms['n_healthy']}) sens {ms['sensitivity']:.4f} "
                  f"spec {ms['specificity']:.4f} bal {ms['balanced_accuracy']:.4f} auc {ms['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
