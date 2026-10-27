import cv2
import numpy as np
import pytest
import torch

from model import INPUT_SIZE, BrainTumorBNCNN, BrainTumorCNN, crop_to_brain, preprocess, preprocess_legacy


def test_crop_all_black_unchanged():
    image = np.zeros((50, 60, 3), np.uint8)
    out = crop_to_brain(image)
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_crop_bright_rectangle_exact():
    image = np.zeros((100, 120, 3), np.uint8)
    image[10:40, 25:90] = 200  # rows 10-39, cols 25-89
    out = crop_to_brain(image)
    assert out.shape == (30, 65, 3)
    assert (out == 200).all()


def test_preprocess_shape_dtype_range_deterministic():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (90, 70, 3), dtype=np.uint8)
    a, b = preprocess(image), preprocess(image)
    assert a.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert a.dtype == torch.float32
    assert a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b)


def test_legacy_preprocess_converts_bgr_to_rgb():
    image = np.zeros((20, 20, 3), np.uint8)
    image[..., 2] = 255  # pure red in BGR
    t = preprocess_legacy(image)[0]
    assert t[0].min() == 1 and t[2].max() == 0


def test_preprocess_is_grey_and_pads_to_square():
    image = np.zeros((100, 40, 3), np.uint8)
    image[:, :, 2] = 200  # coloured, tall and thin
    t = preprocess(image)[0]
    assert torch.equal(t[0], t[1]) and torch.equal(t[1], t[2])  # grey copied to 3 channels
    assert t[0, INPUT_SIZE // 2, 0] == 0  # padded side stays black


def test_preprocess_removes_global_brightness():
    # The shortcut used brightness; a darker copy of a scan must give (almost) the same input
    rng = np.random.default_rng(1)
    image = rng.integers(60, 200, (80, 80, 3), dtype=np.uint8)
    darker = (image * 0.6).astype(np.uint8)
    diff = (preprocess(image) - preprocess(darker)).abs()
    # uint8 rounding of the darker copy allows small per-pixel differences
    assert diff.mean() < 2 / 255 and diff.max() <= 8 / 255  # measured: mean 1.5/255, raw inputs differ by 52/255
    assert (torch.from_numpy(image).float() - torch.from_numpy(darker).float()).abs().mean() > 40  # raw inputs differ a lot


@pytest.mark.parametrize('net', [BrainTumorBNCNN, BrainTumorCNN])
def test_logits_shape_and_softmax(net):
    torch.manual_seed(0)
    model = net().eval()
    with torch.no_grad():
        logits = model(torch.rand(4, 3, INPUT_SIZE, INPUT_SIZE))
    assert logits.shape == (4, 2)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
