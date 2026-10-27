"""OpenCV image-quality inspector for brain MRI slices.

Deterministic measurements plus warnings against configurable thresholds. The
thresholds come from the training images' own distribution (see
experiments/quality/calibrate.py): a warning means "unusual compared with what the
model was trained on", not a clinically validated quality judgement.
"""
import cv2
import numpy as np

from model import BACKGROUND_THRESHOLD, crop_to_brain

# Every threshold the warnings use. Each is the 1st or 99th percentile of the
# leak-free training split (n=1,916; experiments/quality/calibration.json), rounded
# outward, so about 1 in 100 training images trips each check.
QUALITY_THRESHOLDS = {
    'min_sharpness': 80.0,          # variance of the Laplacian, crop resized to 256 px (train p1 87)
    'min_brightness': 50.0,         # mean grey level inside the brain mask, 0-255 (p1 57)
    'max_brightness': 190.0,        # (p99 181)
    'max_clipped_fraction': 0.25,   # share of brain pixels at >= 250, saturated (p99 0.254)
    'min_contrast': 22.0,           # standard deviation inside the brain mask (p1 23.1)
    'min_crop_coverage': 0.40,      # largest region's share of the crop box (p1 0.399)
    'min_crop_side_fraction': 0.50, # crop side / image side; smaller = over-aggressive crop (p1 0.536)
    'max_background_fraction': 0.80,  # (p99 0.757)
    'min_resolution': 128,          # shorter side in pixels; below this the model upscales (p1 173)
    'max_colour_spread': 10.0,      # mean channel deviation; MRI slices are grey (p99 1.7)
    'max_stray_regions': 6,         # bright regions outside the head: text, labels (p99 6)
}

ANALYSIS_SIZE = 256  # sharpness is measured at a fixed size so it doesn't scale with resolution


def _brain_mask(gray):
    return (gray > BACKGROUND_THRESHOLD).astype(np.uint8)


def _regions(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return count, labels, stats  # stats[0] is the background


def measure(image_bgr):
    """All raw measurements, no judgement."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = _brain_mask(gray)
    brain = gray[mask > 0]

    crop = crop_to_brain(image_bgr)
    ch, cw = crop.shape[:2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = ANALYSIS_SIZE / max(ch, cw)
    sized = cv2.resize(crop_gray, (max(1, round(cw * scale)), max(1, round(ch * scale))),
                       interpolation=cv2.INTER_AREA)

    count, _, stats = _regions(mask)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], int)
    largest = int(areas.max()) if areas.size else 0
    # Bright regions away from the head bigger than 0.1% of the image: text, labels, rulers
    stray = int(((areas > 0.001 * h * w) & (areas < largest)).sum()) if areas.size else 0

    channels = image_bgr.astype(np.int16)
    return {
        'width': int(w),
        'height': int(h),
        'sharpness': float(cv2.Laplacian(sized, cv2.CV_64F).var()),
        'brightness': float(brain.mean()) if brain.size else 0.0,
        'clipped_fraction': float((brain >= 250).mean()) if brain.size else 0.0,
        'contrast': float(brain.std()) if brain.size else 0.0,
        'dynamic_range': float(np.percentile(brain, 99) - np.percentile(brain, 1)) if brain.size else 0.0,
        'crop_width': int(cw),
        'crop_height': int(ch),
        'crop_area_fraction': float(cw * ch / (w * h)),
        'crop_coverage': float(largest / (cw * ch)) if largest else 0.0,
        'background_fraction': float(1 - mask.mean()),
        'stray_regions': stray,
        'colour_spread': float(np.abs(channels - channels.mean(axis=2, keepdims=True)).mean()),
    }


def assess(image_bgr, thresholds=QUALITY_THRESHOLDS):
    """Measurements plus one check per quality aspect, ready for the API."""
    m = measure(image_bgr)
    t = thresholds
    empty = m['crop_coverage'] == 0

    checks = [
        _check('sharpness', 'Sharpness', m['sharpness'] < t['min_sharpness'],
               f"{m['sharpness']:.0f}", f">= {t['min_sharpness']:.0f}",
               'Image looks blurry (low edge detail).'),
        _check('brightness', 'Brightness',
               not t['min_brightness'] <= m['brightness'] <= t['max_brightness'],
               f"{m['brightness']:.0f}", f"{t['min_brightness']:.0f}-{t['max_brightness']:.0f}",
               'Unusually dark scan.' if m['brightness'] < t['min_brightness'] else 'Unusually bright scan.'),
        _check('clipping', 'Saturated pixels', m['clipped_fraction'] > t['max_clipped_fraction'],
               f"{m['clipped_fraction']:.1%}", f"<= {t['max_clipped_fraction']:.0%}",
               'Many pixels are at maximum brightness, so detail there is lost.'),
        _check('contrast', 'Contrast', m['contrast'] < t['min_contrast'],
               f"{m['contrast']:.0f}", f">= {t['min_contrast']:.0f}",
               'Low contrast between tissues.'),
        _check('crop', 'Brain crop',
               empty or m['crop_coverage'] < t['min_crop_coverage']
               or min(m['crop_width'] / m['width'], m['crop_height'] / m['height']) < t['min_crop_side_fraction'],
               f"{m['crop_coverage']:.0%} filled", f">= {t['min_crop_coverage']:.0%}",
               'No head found in the image.' if empty else
               'The crop box is stretched by something outside the head, or is too tight.'),
        _check('background', 'Background', m['background_fraction'] > t['max_background_fraction'],
               f"{m['background_fraction']:.0%}", f"<= {t['max_background_fraction']:.0%}",
               'Mostly empty background around a small head.'),
        _check('resolution', 'Resolution', min(m['width'], m['height']) < t['min_resolution'],
               f"{m['width']}×{m['height']}", f"shorter side >= {t['min_resolution']}",
               'Image is smaller than the model input, so it gets upscaled.'),
        _check('artefacts', 'Stray regions', m['stray_regions'] > t['max_stray_regions'],
               str(m['stray_regions']), f"<= {t['max_stray_regions']}",
               'Bright marks outside the head (text, labels or borders).'),
        _check('colour', 'Greyscale', m['colour_spread'] > t['max_colour_spread'],
               f"{m['colour_spread']:.1f}", f"<= {t['max_colour_spread']:.0f}",
               'Image has colour; MRI slices are greyscale, so this may not be an MRI.'),
    ]
    return {
        'metrics': m,
        'checks': checks,
        'warnings': [c['message'] for c in checks if c['status'] == 'warning'],
    }


def _check(key, label, failed, value, expected, message):
    return {
        'key': key,
        'label': label,
        'value': value,
        'expected': expected,
        'status': 'warning' if failed else 'ok',
        'message': message if failed else '',
    }
