import base64
import os
import sys
import threading
from pathlib import Path

# Before cv2 is imported anywhere: refuse to decode images over 50 megapixels, so a
# small but huge-dimension PNG ("decompression bomb") can't exhaust memory
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(50_000_000))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import reliability  # noqa: E402
from gradcam import heatmap_image  # noqa: E402
from model import CLASS_NAMES, INPUT_SIZE, crop_to_brain, load_model, pad_to_square  # noqa: E402
from quality_analysis import QUALITY_THRESHOLDS  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DISPLAY_SIZE = 512
ALLOWED_TYPES = {'image/jpeg', 'image/png'}

model = load_model()

# ReLU of the 4th conv block (16x16 map). The 5th block's map is only 4x4 at
# 128 px input, too coarse to show anything useful.
GRAD_CAM_LAYER = model.features[3][2]

# Grad-CAM registers hooks on the one shared model; two requests at once would mix
# their activations, so analyses run one at a time (in FastAPI's worker threads)
ANALYSIS_LOCK = threading.Lock()

app = FastAPI(title="MRI VisionGuard")
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


def encode_png(image):
    ok, buffer = cv2.imencode('.png', image)
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode()


def fit(image, size=MAX_DISPLAY_SIZE):
    h, w = image.shape[:2]
    scale = min(1.0, size / max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


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
        'quality_thresholds': QUALITY_THRESHOLDS,
        'reliability_thresholds': reliability.RELIABILITY_THRESHOLDS,
    }


@app.post('/api/predict')
def predict(file: UploadFile = File(...)):
    # Plain def: FastAPI runs it in a worker thread, so CPU work doesn't block the server
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Please upload a JPEG or PNG image.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")

    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        # OpenCV 5 raises (rather than returning None) for images over the pixel limit
        raise HTTPException(413, "Image dimensions are too large.")
    if image is None:
        raise HTTPException(400, "Could not read that file as an image.")

    with ANALYSIS_LOCK:
        result = reliability.run(image, model, GRAD_CAM_LAYER)
    p = result['tumor_probability']
    probabilities = {'No tumor': 1 - p, 'Tumor': p}
    prediction = CLASS_NAMES[result['label']]

    display = fit(pad_to_square(crop_to_brain(image)))
    model_view = (result['tensor'][0, 0].numpy() * 255).round().astype(np.uint8)
    quality = result['quality']

    return {
        'prediction': prediction,
        'confidence': probabilities[prediction],
        'probabilities': probabilities,
        'original': encode_png(fit(image)),
        'image': encode_png(display),
        'heatmap': encode_png(heatmap_image(result['cam'], display.shape[1], display.shape[0])),
        'model_view': encode_png(model_view),
        'quality': {'checks': quality['checks'], 'metrics': quality['metrics']},
        'reliability': result['reliability'],
        'warnings': quality['warnings'],
    }
