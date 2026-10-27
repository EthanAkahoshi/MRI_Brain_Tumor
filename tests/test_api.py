"""Upload validation and response shape of the web API."""
import cv2
import numpy as np
import pytest

from model import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="final weights missing")


@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    from webapp.app import app
    with TestClient(app) as c:
        yield c


def png_bytes(image):
    return cv2.imencode('.png', image)[1].tobytes()


def test_rejects_wrong_type(client):
    r = client.post('/api/predict', files={'file': ('a.txt', b'hello', 'text/plain')})
    assert r.status_code == 415


def test_rejects_malformed_image(client):
    r = client.post('/api/predict', files={'file': ('a.png', b'not an image', 'image/png')})
    assert r.status_code == 400


def test_rejects_oversized_upload(client):
    big = b'\x89PNG' + b'0' * (10 * 1024 * 1024 + 10)
    r = client.post('/api/predict', files={'file': ('a.png', big, 'image/png')})
    assert r.status_code == 413


def test_predict_returns_quality_and_reliability(client):
    from tests.test_quality import head
    r = client.post('/api/predict', files={'file': ('h.png', png_bytes(head()), 'image/png')})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['prediction'] in ('Tumor', 'No tumor')
    assert sum(body['probabilities'].values()) == pytest.approx(1.0)
    assert len(body['quality']['checks']) == 9
    assert body['reliability']['status'] in ('consistent', 'sensitive', 'poor_quality', 'review')
    assert body['reliability']['trace'][0]['step'] == 'Initial analysis'
    for key in ('original', 'image', 'heatmap', 'model_view'):
        assert body[key].startswith('data:image/png;base64,')


def test_model_info_exposes_thresholds(client):
    info = client.get('/api/model').json()
    assert info['input_size'] == 128
    assert 'min_sharpness' in info['quality_thresholds']
    assert 'max_score_change' in info['reliability_thresholds']
