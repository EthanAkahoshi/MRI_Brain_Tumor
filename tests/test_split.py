import csv
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / 'experiments' / 'leak_free_split' / 'split.csv'
HELD_OUT = [ROOT / 'datasets' / 'Testing',
            Path(r'Z:\GoogleDownloads\Brain\pred\brain_tumor_dataset')]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


@pytest.fixture(scope='module')
def rows():
    if not SPLIT.exists():
        pytest.skip(f"{SPLIT} not written yet (the Investigator is building it)")
    with open(SPLIT, newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows, "split.csv is empty"
    assert {'path', 'md5', 'source', 'label', 'group', 'split'} <= rows[0].keys()
    assert {r['split'] for r in rows} <= {'train', 'val', 'test'}
    return rows


def splits_per(rows, key):
    seen = {}
    for r in rows:
        seen.setdefault(r[key], set()).add(r['split'])
    return {k: v for k, v in seen.items() if len(v) > 1}


@pytest.mark.parametrize('key', ['path', 'md5', 'group'])
def test_no_key_in_two_splits(rows, key):
    shared = splits_per(rows, key)
    assert not shared, f"{len(shared)} {key}s in more than one split, e.g. {list(shared.items())[:3]}"


def test_md5_column_is_correct_sample(rows):
    for r in rows[::max(1, len(rows) // 20)]:
        path = Path(r['path'])
        path = path if path.is_absolute() else ROOT / path
        assert hashlib.md5(path.read_bytes()).hexdigest() == r['md5'], r['path']


def test_no_overlap_with_testing_or_external(rows):
    split_md5 = {r['md5'] for r in rows}
    held_out = {}
    for folder in HELD_OUT:
        assert folder.exists(), folder
        for p in folder.rglob('*'):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                held_out[hashlib.md5(p.read_bytes()).hexdigest()] = p
    leaked = split_md5 & held_out.keys()
    assert not leaked, f"{len(leaked)} split images are byte-identical to held-out ones, e.g. {[str(held_out[m]) for m in list(leaked)[:3]]}"
