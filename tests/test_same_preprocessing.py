"""mainTest.py, the web app and a direct preprocess()+model call must agree.

mainTest.py runs predictions at import time, so it is never imported. Instead:
  1. load_image() is pulled out of its source with `ast` and run on its own,
     so we test the exact code it ships, not a copy of it;
  2. the whole script is run in a subprocess on one image and its printed
     probability is compared with the direct one.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import model as model_module
from model import CLASS_NAMES, DECISION_THRESHOLD, MODEL_PATH, load_model, preprocess, tumor_probability

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = MODEL_PATH
SAMPLE = ROOT / 'datasets' / 'yes' / 'glioma' / 'Tr-gl_1.jpg'

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.exists() and SAMPLE.exists()), reason="weights or sample image missing")


@pytest.fixture(scope='module')
def net():
    return load_model()


@pytest.fixture(scope='module')
def direct(net):
    tensor = preprocess(cv2.imread(str(SAMPLE)))
    return tensor, tumor_probability(net, tensor)


def test_maintest_load_image_matches_preprocess(direct):
    tree = ast.parse((ROOT / 'mainTest.py').read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_image')
    namespace = {'cv2': cv2, 'preprocess': model_module.preprocess}
    exec(compile(ast.Module([func], []), 'mainTest.py', 'exec'), namespace)
    assert torch.equal(namespace['load_image'](SAMPLE), direct[0])


def test_maintest_script_prints_same_probability(direct):
    out = subprocess.run([sys.executable, 'mainTest.py', str(SAMPLE)], cwd=ROOT,
                         capture_output=True, text=True, timeout=120, check=True).stdout
    match = re.search(rf'{re.escape(SAMPLE.name)}: (.+) \(([\d.]+)%\)', out)
    assert match, out
    p = direct[1]
    predicted = int(p >= DECISION_THRESHOLD)
    assert match.group(1) == CLASS_NAMES[predicted]
    assert float(match.group(2)) == pytest.approx((p if predicted else 1 - p) * 100, abs=0.05)


def test_webapp_predict_matches_direct(direct):
    from fastapi.testclient import TestClient
    from webapp.app import app

    with TestClient(app) as client:
        response = client.post('/api/predict', files={
            'file': (SAMPLE.name, SAMPLE.read_bytes(), 'image/jpeg')})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['probabilities']['Tumor'] == pytest.approx(direct[1], abs=1e-6)
    assert body['prediction'] == CLASS_NAMES[int(direct[1] >= DECISION_THRESHOLD)]


def test_imdecode_equals_imread():
    # The web app decodes bytes; mainTest and training read files. Same pixels?
    decoded = cv2.imdecode(np.frombuffer(SAMPLE.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, cv2.imread(str(SAMPLE)))
