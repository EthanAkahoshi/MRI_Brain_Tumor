import cv2
import numpy as np
import torch


def grad_cam(model, image, target_class, layer):
    # Grad-CAM: weight each feature map of `layer` by how much it pushes the
    # score of `target_class` up, then sum them into one heatmap.
    # https://arxiv.org/abs/1610.02391
    captured = {}

    def save_activation(module, inputs, output):
        captured['activation'] = output
        output.register_hook(lambda grad: captured.__setitem__('gradient', grad))

    handle = layer.register_forward_hook(save_activation)
    try:
        model.zero_grad()
        with torch.enable_grad():
            scores = model(image)
            scores[0, target_class].backward()
    finally:
        handle.remove()

    activation = captured['activation'][0].detach()   # (channels, h, w)
    gradient = captured['gradient'][0]                 # (channels, h, w)

    weights = gradient.mean(dim=(1, 2))                # one weight per feature map
    cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))

    cam = cam.numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam                                         # (h, w), values 0-1


def heatmap_image(cam, width, height):
    # Colour the heatmap and use its strength as transparency, so weak regions
    # (below ~15%) are fully see-through when it is laid over the scan
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)

    colour = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # perceptually smoother than JET, kinder to colour-blind viewers
    alpha = (np.clip((cam - 0.15) / 0.5, 0, 1) * 255).astype(np.uint8)
    return np.dstack([colour, alpha])                  # BGRA
