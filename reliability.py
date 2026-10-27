"""Second-pass reliability check: perception -> decision -> action -> comparison.

Plain rule-based logic, not a learned or autonomous agent:
1. Perception: OpenCV quality measurements, the model's score and its Grad-CAM map.
2. Decision: a fixed rule table picks at most ONE follow-up action (or none).
3. Action: re-process the image with that controlled change and re-run the model once.
4. Comparison: score change, label change and heatmap agreement give a status.

The reported prediction is always the first one. The follow-up only says whether
it held up; it never replaces it (that would be tuning the answer at test time).
"""
import cv2
import numpy as np

from gradcam import grad_cam
from model import DECISION_THRESHOLD, preprocess, tumor_probability
from quality_analysis import assess

# Set from the leak-free VALIDATION split (experiments/quality/reliability_calibration.json),
# so roughly 1 in 20 validation scans trips each signal.
RELIABILITY_THRESHOLDS = {
    'low_confidence_margin': 0.25,   # |P(tumor) - threshold| below this = low confidence
    'max_background_heat': 0.55,     # share of Grad-CAM weight outside the head (val p95: 0.38 tumour, 0.69 no tumour)
    'max_score_change': 0.20,        # mirror |delta P(tumor)| above this = sensitive (val p95: 0.13 / 0.30)
    'min_heatmap_agreement': 0.35,   # tumour predictions only (val p5 0.36); "no tumor" maps are diffuse by nature
}

STATUS = {
    'consistent': 'Consistent experimental result',
    'sensitive': 'Preprocessing-sensitive result',
    'poor_quality': 'Poor input quality',
    'review': 'Requires human review',
}

# Checks that a different preprocessing step can't fix
SEVERE_QUALITY = {'sharpness', 'colour', 'resolution'}


def clahe(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def largest_region_crop(image_bgr):
    # Black out everything except the largest bright region, so stray text or
    # border marks can't stretch the crop box
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 20).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return image_bgr
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where((labels == keep)[..., None], image_bgr, 0).astype(image_bgr.dtype)


def mirror(image_bgr):
    return cv2.flip(image_bgr, 1)


# (name, function, what it does). Order = priority when several rules fire.
ACTIONS = {
    'contrast': ('Contrast normalisation', clahe, 'CLAHE on the grey image (OpenCV createCLAHE)'),
    'crop': ('Largest-region crop', largest_region_crop, 'keep only the largest bright region before cropping'),
    'mirror': ('Mirror test', mirror, 'flip left-right; a brain scan should get the same answer'),
}


def head_mask(tensor):
    # Outline of the head in the model's input, holes filled: normalisation turns the
    # darkest tissue inside the brain to 0, and that must not count as background
    mask = (tensor[0, 0].numpy() > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def background_heat(cam, tensor):
    # Share of Grad-CAM weight that falls outside the head in the model's input
    head = head_mask(tensor)
    cam = cv2.resize(cam, head.shape[::-1], interpolation=cv2.INTER_LINEAR)
    total = cam.sum()
    return float(cam[~head].sum() / total) if total > 0 else 0.0


def heatmap_agreement(cam_a, cam_b):
    a, b = cam_a.ravel(), cam_b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def choose_action(warning_keys, low_confidence, heat_outside, t=RELIABILITY_THRESHOLDS):
    """The rule table. Returns (action key or None, reason)."""
    if warning_keys & {'contrast', 'brightness', 'clipping'}:
        return 'contrast', 'quality check flagged brightness or contrast'
    if 'crop' in warning_keys or 'artefacts' in warning_keys:
        return 'crop', 'quality check flagged the crop or stray marks outside the head'
    if low_confidence:
        return 'mirror', 'model confidence is low'
    if heat_outside > t['max_background_heat']:
        return 'mirror', 'an unusually large share of heatmap weight is outside the head'
    return None, 'no reliability rule fired'


def decide_status(warning_keys, low_confidence, followup, t=RELIABILITY_THRESHOLDS):
    """Status from the evidence; pure function so it can be tested on its own."""
    if warning_keys & SEVERE_QUALITY:
        return 'poor_quality'
    sensitive = bool(followup) and (
        followup['label_changed']
        or abs(followup['score_change']) > t['max_score_change']
        # Heatmap agreement only means something when there is a lesion to point at
        or (followup['predicted_tumor'] and followup['heatmap_agreement'] is not None
            and followup['heatmap_agreement'] < t['min_heatmap_agreement']))
    if sensitive:
        return 'review' if low_confidence else 'sensitive'
    if low_confidence:
        return 'review'
    return 'consistent'


def run(image_bgr, model, layer, t=RELIABILITY_THRESHOLDS):
    trace = []

    # 1. Perception
    quality = assess(image_bgr)
    warning_keys = {c['key'] for c in quality['checks'] if c['status'] == 'warning'}
    tensor = preprocess(image_bgr)
    p = tumor_probability(model, tensor)
    label = int(p >= DECISION_THRESHOLD)
    cam = grad_cam(model, tensor, label, layer)
    heat_outside = background_heat(cam, tensor)
    low_confidence = abs(p - DECISION_THRESHOLD) < t['low_confidence_margin']
    trace.append({
        'step': 'Initial analysis',
        'detail': (f"P(tumor) {p:.2f}; {len(warning_keys)} quality warning(s)"
                   f"{' (' + ', '.join(sorted(warning_keys)) + ')' if warning_keys else ''}; "
                   f"{heat_outside:.0%} of heatmap weight outside the head"),
    })

    # 2. Decision
    action, reason = choose_action(warning_keys, low_confidence, heat_outside, t)
    trace.append({'step': 'Decision',
                  'detail': f"{ACTIONS[action][0]} selected: {reason}" if action else f"No follow-up: {reason}"})

    # 3. Action + 4. comparison (at most one extra inference)
    followup = None
    if action:
        name, transform, how = ACTIONS[action]
        tensor2 = preprocess(transform(image_bgr))
        p2 = tumor_probability(model, tensor2)
        agreement = None
        if action == 'mirror':
            # Only the mirror test keeps the same picture area, so only there can the
            # two heatmaps be compared cell by cell (the crop and CLAHE move the crop box)
            cam2 = grad_cam(model, tensor2, label, layer)[:, ::-1]  # flip back to line up
            agreement = heatmap_agreement(cam, cam2)
        followup = {
            'action': name,
            'tool': how,
            'tumor_probability': p2,
            'score_change': p2 - p,
            'label_changed': int(p2 >= DECISION_THRESHOLD) != label,
            'predicted_tumor': bool(label),
            'heatmap_agreement': agreement,
        }
        trace.append({'step': 'Action', 'detail': f"{name}: {how}; re-ran the model once"})
        trace.append({'step': 'Comparison', 'detail': (
            f"P(tumor) {p:.2f} -> {p2:.2f} ({p2 - p:+.2f}); "
            f"label {'changed' if followup['label_changed'] else 'unchanged'}"
            + (f"; heatmap agreement {agreement:.2f}" if agreement is not None else ""))})

    status = decide_status(warning_keys, low_confidence, followup, t)
    review = status != 'consistent'
    trace.append({'step': 'Result', 'detail': f"{STATUS[status]}"
                  f"{'; human review requested' if review else ''}"})

    return {
        'tensor': tensor,
        'tumor_probability': p,
        'label': label,
        'cam': cam,
        'quality': quality,
        'reliability': {
            'status': status,
            'status_label': STATUS[status],
            'human_review': review,
            'low_confidence': low_confidence,
            'background_heat': heat_outside,
            'followup': followup,
            'trace': trace,
        },
    }
