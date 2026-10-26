import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradcam import grad_cam, heatmap_image  # noqa: E402
from model import BrainTumorCNN, CLASS_NAMES, INPUT_SIZE, crop_to_brain, preprocess  # noqa: E402

MODEL_PATH = ROOT / 'BrainTumorBest.pt'
STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DISPLAY_SIZE = 512
ALLOWED_TYPES = {'image/jpeg', 'image/png'}

model = BrainTumorCNN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# ReLU after the last convolution: the most detailed layer that still "knows"
# about tumours, so it is the usual choice for Grad-CAM
GRAD_CAM_LAYER = model.features[7]

app = FastAPI(title="Brain MRI Tumor Classifier")
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


def encode_png(image):
    ok, buffer = cv2.imencode('.png', image)
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode()


def looks_like_grayscale(image_bgr):
    # MRI scans are grayscale; strongly coloured images are probably not MRIs
    channels = image_bgr.astype(np.int16)
    spread = np.abs(channels - channels.mean(axis=2, keepdims=True)).mean()
    return spread < 10


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/model')
def model_info():
    return {
        'classes': CLASS_NAMES,
        'input_size': INPUT_SIZE,
        'parameters': sum(p.numel() for p in model.parameters()),
    }


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Please upload a JPEG or PNG image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not read that file as an image.")

    tensor = preprocess(image)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    predicted = int(probabilities.argmax())

    # The model only sees the cropped brain, so show that crop and stretch the
    # heatmap over it, keeping the crop's aspect ratio
    cropped = crop_to_brain(image)
    height, width = cropped.shape[:2]
    scale = min(1.0, MAX_DISPLAY_SIZE / max(height, width))
    display_w, display_h = round(width * scale), round(height * scale)
    display = cv2.resize(cropped, (display_w, display_h), interpolation=cv2.INTER_AREA)

    cam = grad_cam(model, tensor, predicted, GRAD_CAM_LAYER)
    model_view = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))

    return {
        'prediction': CLASS_NAMES[predicted],
        'confidence': probabilities[predicted].item(),
        'probabilities': {
            name: probabilities[i].item() for i, name in enumerate(CLASS_NAMES)
        },
        'image': encode_png(display),
        'heatmap': encode_png(heatmap_image(cam, display_w, display_h)),
        'model_view': encode_png(model_view),
        'warnings': [] if looks_like_grayscale(image) else [
            "This image is in colour. MRI scans are grayscale, so this may not be an "
            "MRI, and the model will still give an answer that means nothing."
        ],
    }
