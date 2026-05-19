#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tangjinqi/basketball"
REPO_DIR="${ROOT_DIR}/2022-winners-camera-calibration-challenge"
INPUT_IMAGE="${ROOT_DIR}/image.png"
OUTPUT_IMAGE="${ROOT_DIR}/output.png"
MODEL_PATH="${REPO_DIR}/models/model_challenge.pth"

cd "${REPO_DIR}"

"/home/tangjinqi/miniconda3/envs/task/bin/python" - <<'PY'
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT_DIR = "/home/tangjinqi/basketball"
REPO_DIR = os.path.join(ROOT_DIR, "2022-winners-camera-calibration-challenge")
INPUT_IMAGE = os.path.join(ROOT_DIR, "image.png")
OUTPUT_IMAGE = os.path.join(ROOT_DIR, "output.png")
MODEL_PATH = os.path.join(REPO_DIR, "models", "model_challenge.pth")

sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)

from data.datasets.viewds import getFieldPoints
from kalicalib.estimateHomography import estimateCalibHM, getFieldPoints2d
from kalicalib.model_resnet import makeModel

IMG_WIDTH = 960
IMG_HEIGHT = 540

if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Challenge model not found: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = makeModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

image_rgb = Image.open(INPUT_IMAGE).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
image_rgb_np = np.array(image_rgb)

tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

image_tensor = tf(image_rgb_np).to(device).unsqueeze(0)

with torch.no_grad():
    heatmaps = model(image_tensor)

# Existing visualization code draws with OpenCV colors, so use BGR here.
visualization_bgr = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)
estimateCalibHM(
    heatmaps,
    visualization_bgr,
    getFieldPoints2d(),
    getFieldPoints(),
    IMG_HEIGHT,
    IMG_WIDTH,
    False,
)

cv2.imwrite(OUTPUT_IMAGE, visualization_bgr)
print(f"Saved visualization to {OUTPUT_IMAGE}")
PY
