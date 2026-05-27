"""
Crop Disease Detection — FastAPI Backend

Endpoints:
    GET  /health       — Liveness check
    GET  /models       — List available models + metadata
    POST /predict      — Single image prediction
    POST /explain      — Prediction + GradCAM heatmap
"""

import io
import time
import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.inference import InferenceService, MODEL_METADATA

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───
CHECKPOINTS_DIR = "results"
LABEL_MAPPING_PATH = "configs/label_mapping.json"
MAX_IMAGE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# ─── Rate Limiter ───
limiter = Limiter(key_func=get_remote_address)

# ─── Inference Service (global) ───
inference_service: InferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    global inference_service
    logger.info("Starting model loading...")
    inference_service = InferenceService(
        checkpoints_dir=CHECKPOINTS_DIR,
        label_mapping_path=LABEL_MAPPING_PATH,
    )
    inference_service.load_all_models()
    logger.info("API ready.")
    yield
    logger.info("Shutting down...")


# ─── App ───
app = FastAPI(
    title="Crop Disease Detection API",
    description="Multi-model crop disease classification with GradCAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Middleware ───
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ───
def validate_image(file: UploadFile) -> None:
    """Validate uploaded file is an allowed image type and within size limit."""
    # Check extension
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")


async def read_image(file: UploadFile) -> Image.Image:
    """Read uploaded file into PIL Image."""
    contents = await file.read()

    # Check size
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({size_mb:.1f}MB). Max: {MAX_IMAGE_SIZE_MB}MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    return image


# ─── Endpoints ───

@app.get("/health")
async def health_check():
    """Liveness probe for Cloud Run."""
    loaded_models = list(inference_service.models.keys()) if inference_service else []
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "device": str(inference_service.device) if inference_service else "N/A",
    }


@app.get("/models")
async def list_models():
    """List all available models with metadata and performance metrics."""
    if not inference_service:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return {"models": inference_service.get_available_models()}


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    image: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    model_name: str = Form(default="resnet_50", description="Model to use for prediction"),
    top_k: int = Form(default=5, description="Number of top predictions to return"),
):
    """
    Predict crop disease from a leaf image.

    - Upload an image of a crop leaf
    - Optionally specify which model to use
    - Returns predicted disease class with confidence scores
    """
    # Validate
    validate_image(image)

    if model_name not in MODEL_METADATA:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {list(MODEL_METADATA.keys())}",
        )

    if inference_service and model_name not in inference_service.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' checkpoint not available on this server.",
        )

    # Read image
    img = await read_image(image)

    # Predict
    start_time = time.time()
    result = inference_service.predict(img, model_name=model_name, top_k=top_k)
    latency_ms = (time.time() - start_time) * 1000

    # Log
    logger.info(
        f"Prediction: {result['prediction']} ({result['confidence']:.2%}) "
        f"| model={model_name} | latency={latency_ms:.0f}ms"
    )

    return {
        **result,
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/explain")
@limiter.limit("10/minute")
async def explain(
    request: Request,
    image: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    model_name: str = Form(default="resnet_50", description="Model to use"),
):
    """
    Predict + generate GradCAM heatmap overlay.

    Returns prediction results plus a base64-encoded heatmap image.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Validate
    validate_image(image)

    if model_name not in MODEL_METADATA:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'.")

    if inference_service and model_name not in inference_service.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not available.")

    # Read image
    img = await read_image(image)

    # Predict
    start_time = time.time()
    result = inference_service.predict(img, model_name=model_name, top_k=5)

    # GradCAM
    grayscale_cam = inference_service.get_gradcam(img, model_name=model_name)
    latency_ms = (time.time() - start_time) * 1000

    # Generate overlay image
    heatmap_b64 = None
    if grayscale_cam is not None:
        # Prepare RGB image (0-1 range, 224x224)
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        # Center crop to 224
        h, w = img_array.shape[:2]
        top = (h - 224) // 2
        left = (w - 224) // 2
        img_cropped = img_array[top:top+224, left:left+224]

        cam_image = show_cam_on_image(img_cropped, grayscale_cam, use_rgb=True)

        # Encode as base64 PNG
        cam_pil = Image.fromarray(cam_image)
        buffer = io.BytesIO()
        cam_pil.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    logger.info(
        f"Explain: {result['prediction']} ({result['confidence']:.2%}) "
        f"| model={model_name} | latency={latency_ms:.0f}ms"
    )

    return {
        **result,
        "heatmap_base64": heatmap_b64,
        "latency_ms": round(latency_ms, 1),
    }


# ─── Error Handlers ───

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
