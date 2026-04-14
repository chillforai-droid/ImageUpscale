import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import torch

from realesrgan import RealESRGANer
from gfpgan import GFPGANer

app = FastAPI()

# =========================
# SETTINGS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

TILE = 200
TILE_PAD = 10

print("Loading AI models...")

# RealESRGAN
realesrgan = RealESRGANer(
    scale=4,
    model_path=None,
    model=None,
    tile=TILE,
    tile_pad=TILE_PAD,
    pre_pad=0,
    half=False,
    device=device
)

# GFPGAN
gfpgan = GFPGANer(
    model_path=None,
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=realesrgan
)

print("Models loaded successfully!")

# =========================
# API
# =========================

@app.get("/")
def home():
    return {"message": "AI Upscaler Running 🚀"}

@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # resize (memory safe)
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # AI Enhance
        _, _, output = gfpgan.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # compress output
        _, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}
