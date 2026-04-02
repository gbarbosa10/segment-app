import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_ROOT / "static"
MODEL_PATH = APP_ROOT / "app" / "FCN_best_train_dice.ckpt"
application = FastAPI(title="Image Segmentation Demo")

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # Avoid loading an old torchvision release through torch.hub; it is
        # incompatible with modern torch/onnx internals.
        self.backbone = fcn_resnet50(weights=None, weights_backbone=None).backbone

        dec0conv1 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        dec0norm1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        dec0relu1 = nn.ReLU(inplace=True)
        dec0conv2 = nn.Dropout(p=0.1, inplace=False)
        dec0norm2 = nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder0 = nn.Sequential(dec0conv1, dec0norm1, dec0relu1, dec0conv2, dec0norm2)
        self.conv_back_ground = nn.Conv2d(21, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        backbone_out = self.backbone(x)
        upsample_im = self.upsample_1(backbone_out['out'])
        upsample_im2 = self.upsample_2(upsample_im)
        upsample_im3 = self.upsample_2(upsample_im2)

        encodbg = self.decoder0(upsample_im3)
        bg_out = self.conv_back_ground(encodbg)

        return bg_out

def load_model():
    torch.manual_seed(0)
    model = FCN()
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state_dict = {key.replace("model.", ""): value for key, value in state['state_dict'].items()}
        model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image: Image.Image, size=224):
    image = image.convert("RGB")
    image = image.resize((size, size))
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def mask_to_image(mask_array: np.ndarray) -> Image.Image:
    mask = (mask_array > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (220, 40, 40, 140))
    overlay = Image.composite(red, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


model = load_model()


@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return JSONResponse({"error": "Unable to read image."}, status_code=400)

    resized, tensor = preprocess_image(image, size= 224)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask_image = mask_to_image(prob)
    overlay_image = overlay_mask(resized, mask_image)

    return {
        "original": image_to_data_url(resized),
        "mask": image_to_data_url(mask_image),
        "overlay": image_to_data_url(overlay_image),
    }

application.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
