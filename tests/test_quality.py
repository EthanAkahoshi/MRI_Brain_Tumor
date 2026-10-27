import cv2
import numpy as np

from quality_analysis import QUALITY_THRESHOLDS, assess, measure


def head(size=200, value=120, noise=40, seed=0):
    # A synthetic "head": a textured ellipse on black, like an axial slice
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 - 15), 0, 0, 360, value, -1)
    texture = rng.integers(-noise, noise + 1, img.shape)
    img = np.where(img > 0, np.clip(img.astype(int) + texture, 21, 255), 0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def warned(image, key):
    return next(c for c in assess(image)['checks'] if c['key'] == key)['status'] == 'warning'


def test_clean_image_has_no_warnings():
    assert assess(head())['warnings'] == []


def test_measurements_are_deterministic():
    assert measure(head()) == measure(head())


def test_blur_lowers_sharpness_and_warns():
    sharp = head()
    blurry = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert measure(blurry)['sharpness'] < measure(sharp)['sharpness']
    assert warned(blurry, 'sharpness') and not warned(sharp, 'sharpness')


def test_dark_and_bright_scans_warn():
    assert warned(head(value=35, noise=5), 'brightness')
    assert warned(head(value=230, noise=5), 'brightness')


def test_saturated_scan_warns_on_clipping():
    assert warned(head(value=252, noise=2), 'clipping')


def test_flat_scan_warns_on_low_contrast():
    assert warned(head(noise=2), 'contrast')


def test_stray_text_outside_head_warns():
    image = head(size=300)  # head spans x 50-250, so a column at x < 40 stays outside it
    for i in range(QUALITY_THRESHOLDS['max_stray_regions'] + 1):
        cv2.putText(image, 'R', (4, 40 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    assert warned(image, 'artefacts')


def test_empty_image_fails_crop_check():
    check = next(c for c in assess(np.zeros((150, 150, 3), np.uint8))['checks'] if c['key'] == 'crop')
    assert check['status'] == 'warning' and 'No head' in check['message']


def test_colour_image_warns():
    image = head()
    image[..., 2] = 255
    assert warned(image, 'colour')


def test_small_image_warns_on_resolution():
    assert warned(cv2.resize(head(), (90, 90)), 'resolution')


def test_custom_thresholds_are_respected():
    strict = dict(QUALITY_THRESHOLDS, min_sharpness=1e12)
    assert any(c['key'] == 'sharpness' and c['status'] == 'warning'
               for c in assess(head(), strict)['checks'])
