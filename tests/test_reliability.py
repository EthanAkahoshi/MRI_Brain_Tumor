import numpy as np
import pytest
import torch

import reliability
from model import load_model, MODEL_PATH
from reliability import choose_action, decide_status, heatmap_agreement


def followup(score_change=0.0, label_changed=False, agreement=0.9, tumour=True):
    return {'score_change': score_change, 'label_changed': label_changed,
            'heatmap_agreement': agreement, 'predicted_tumor': tumour}


@pytest.mark.parametrize('warnings, low_conf, heat, expected', [
    ({'contrast'}, False, 0.1, 'contrast'),
    ({'brightness', 'crop'}, False, 0.1, 'contrast'),  # contrast fix has priority
    ({'crop'}, False, 0.1, 'crop'),
    ({'artefacts'}, False, 0.1, 'crop'),
    (set(), True, 0.1, 'mirror'),
    (set(), False, 0.9, 'mirror'),
    (set(), False, 0.1, None),  # nothing fired: no extra inference
])
def test_rule_table(warnings, low_conf, heat, expected):
    assert choose_action(warnings, low_conf, heat)[0] == expected


def test_status_rules():
    assert decide_status({'sharpness'}, False, None) == 'poor_quality'
    assert decide_status(set(), False, None) == 'consistent'
    assert decide_status(set(), True, None) == 'review'
    assert decide_status(set(), False, followup()) == 'consistent'
    assert decide_status(set(), False, followup(label_changed=True)) == 'sensitive'
    assert decide_status(set(), False, followup(score_change=-0.5)) == 'sensitive'
    assert decide_status(set(), True, followup(score_change=0.5)) == 'review'
    # Heatmap disagreement only counts for tumour predictions
    assert decide_status(set(), False, followup(agreement=0.0)) == 'sensitive'
    assert decide_status(set(), False, followup(agreement=0.0, tumour=False)) == 'consistent'


def test_crop_and_contrast_followups_skip_heatmap_agreement():
    # Those follow-ups change the crop box, so their heatmaps can't be compared cell by cell
    assert decide_status(set(), False, followup(agreement=None)) == 'consistent'


def test_low_resolution_is_poor_quality():
    assert decide_status({'resolution'}, False, None) == 'poor_quality'


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="final weights missing")
def test_stray_mark_crop_followup_has_no_agreement():
    import cv2
    from tests.test_quality import head
    image = head(size=300)
    for i in range(8):  # stray letters outside the head trip the crop/artefact rule
        cv2.putText(image, 'L', (4, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    model = load_model()
    rel = reliability.run(image, model, model.features[3][2])['reliability']
    assert rel['followup']['action'] == 'Largest-region crop'
    assert rel['followup']['heatmap_agreement'] is None


def test_heatmap_agreement_bounds():
    a = np.random.default_rng(0).random((16, 16))
    assert heatmap_agreement(a, a) == pytest.approx(1.0)
    assert heatmap_agreement(a, -a) == pytest.approx(-1.0)


def test_background_heat_ignores_dark_tissue_inside_head():
    # A ring-shaped head (dark centre) with all heat in the centre: that's inside the head
    tensor = torch.zeros(1, 3, 32, 32)
    yy, xx = np.mgrid[:32, :32]
    r = np.hypot(yy - 16, xx - 16)
    tensor[0, 0] = torch.from_numpy(((r > 5) & (r < 12)).astype(np.float32))
    cam = (r < 5).astype(np.float32)
    assert reliability.background_heat(cam, tensor) == pytest.approx(0.0, abs=0.05)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="final weights missing")
def test_run_returns_trace_and_at_most_one_followup():
    from tests.test_quality import head
    model = load_model()
    out = reliability.run(head(), model, model.features[3][2])
    rel = out['reliability']
    assert rel['status'] in reliability.STATUS
    steps = [s['step'] for s in rel['trace']]
    assert steps[0] == 'Initial analysis' and steps[-1] == 'Result'
    assert steps.count('Action') == (1 if rel['followup'] else 0)
    assert 0 <= out['tumor_probability'] <= 1
