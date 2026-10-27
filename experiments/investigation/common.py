# File listing shared by the investigation scripts. Mirrors mainTrain.py's
# ordering (os.walk, sorted) so the random split can be reproduced exactly.
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'datasets')
EXTERNAL = r'Z:\GoogleDownloads\Brain\pred\brain_tumor_dataset'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def image_files(folder):
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, name)


def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, '/')


def pool_files():
    # (path, label, source) in mainTrain.py order. Top-level files of no/ and
    # yes/ are the original dataset; files in subfolders are the new Kaggle set.
    out = []
    for folder, label in [('no', 0), ('yes', 1)]:
        base = os.path.join(DATA, folder)
        for p in image_files(base):
            out.append((p, label, 'original' if os.path.dirname(p) == base else 'new'))
    return out


def benchmark_files():
    return [(p, 0 if os.path.basename(os.path.dirname(p)) == 'notumor' else 1, 'testing')
            for p in image_files(os.path.join(DATA, 'Testing'))]


def external_files():
    return [(p, label, 'external') for sub, label in [('no', 0), ('yes', 1)]
            for p in image_files(os.path.join(EXTERNAL, sub))]
