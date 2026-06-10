"""
Crop Disease Detection — FastAPI Backend

Endpoints:
    GET  /health       — Liveness check
    GET  /models       — List available models + metadata
    POST /predict      — Single image prediction (supports A/B testing)
    POST /explain      — Prediction + GradCAM heatmap (supports A/B testing)
"""

import io
import os
import time
import base64
import random
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.inference import InferenceService, MODEL_METADATA
from api.download_checkpoints import download_checkpoints

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───
CHECKPOINTS_DIR = "results"
LABEL_MAPPING_PATH = "configs/label_mapping.json"
MAX_IMAGE_SIZE_MB = 10

# ─── A/B Testing Config ───
AB_TEST_MODEL_A = "resnet_50"       # Control group
AB_TEST_MODEL_B = "mobilenet_v3"    # Treatment group
AB_TEST_SPLIT = 0.5                  # 50/50 split
AB_TEST_WANDB_PROJECT = "crop-disease-detection"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# ─── Rate Limiter ───
limiter = Limiter(key_func=get_remote_address)

# ─── Inference Service (global) ───
inference_service: InferenceService = None
_load_error: str = None

# ─── W&B A/B Test Run (lazy-initialized) ───
_wandb_ab_run = None
_wandb_lock = threading.Lock()


def _load_models_bg():
    """Background thread to download + load model checkpoints."""
    global inference_service, _load_error
    _load_error = None
    print("[_load_models_bg] Thread started", flush=True)
    try:
        print("[_load_models_bg] Calling download_checkpoints()...", flush=True)
        download_checkpoints()
        print("[_load_models_bg] Download finished.", flush=True)

        print("[_load_models_bg] Creating InferenceService...", flush=True)
        inference_service = InferenceService(
            checkpoints_dir=CHECKPOINTS_DIR,
            label_mapping_path=LABEL_MAPPING_PATH,
        )
        print("[_load_models_bg] Loading all models...", flush=True)
        inference_service.load_all_models()
        print(f"[_load_models_bg] Done. Loaded {len(inference_service.models)} models.", flush=True)

        if len(inference_service.models) == 0:
            _load_error = "No models loaded — checkpoints may be missing or corrupted."
            print(f"[_load_models_bg] WARNING: {_load_error}", flush=True)
    except Exception as e:
        _load_error = str(e)
        print(f"[_load_models_bg] FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start server immediately; load models in background."""
    logger.info("Starting API server...")
    thread = threading.Thread(target=_load_models_bg, daemon=True)
    thread.start()
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


def resolve_ab_test(model_name: str) -> tuple:
    """
    If model_name is 'ab_test', randomly select between Model A and Model B.

    Returns:
        (resolved_model_name, is_ab_test, ab_group)
    """
    if model_name == "ab_test":
        if random.random() < AB_TEST_SPLIT:
            return AB_TEST_MODEL_A, True, "control"
        else:
            return AB_TEST_MODEL_B, True, "treatment"
    return model_name, False, None


def log_ab_telemetry(
    model_used: str,
    ab_group: str,
    prediction: str,
    confidence: float,
    latency_ms: float,
):
    """
    Log A/B test telemetry to W&B in the background.
    Called via FastAPI BackgroundTasks so it doesn't block the response.
    """
    global _wandb_ab_run
    try:
        import wandb

        with _wandb_lock:
            if _wandb_ab_run is None:
                # Load .env for WANDB_API_KEY
                env_file = Path(__file__).resolve().parent.parent / ".env"
                if env_file.exists():
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, value = line.split("=", 1)
                                os.environ.setdefault(key.strip(), value.strip())

                _wandb_ab_run = wandb.init(
                    project=AB_TEST_WANDB_PROJECT,
                    name="online-ab-test",
                    tags=["online-ab-test", "production"],
                    job_type="online-ab-test",
                    config={
                        "model_a": AB_TEST_MODEL_A,
                        "model_b": AB_TEST_MODEL_B,
                        "split": AB_TEST_SPLIT,
                    },
                    resume="allow",
                )

        _wandb_ab_run.log({
            "model_used": model_used,
            "ab_group": ab_group,
            "confidence": confidence,
            "latency_ms": latency_ms,
            f"latency/{model_used}": latency_ms,
            f"confidence/{model_used}": confidence,
        })

    except Exception as e:
        logger.warning(f"A/B telemetry logging failed: {e}")


# ─── Endpoints ───

@app.get("/health")
async def health_check():
    """Liveness probe for Cloud Run."""
    loaded_models = list(inference_service.models.keys()) if inference_service else []
    status = "healthy" if len(loaded_models) > 0 else ("error" if _load_error else "loading")
    return {
        "status": status,
        "models_loaded": len(loaded_models),
        "models": loaded_models,
        "device": str(inference_service.device) if inference_service else "N/A",
        "error": _load_error,
    }


@app.get("/models")
async def list_models():
    """List all available models with metadata and performance metrics."""
    if _load_error and not inference_service:
        raise HTTPException(status_code=503, detail=f"Service failed to start: {_load_error}")
    if not inference_service:
        raise HTTPException(status_code=503, detail="Service is still loading models. Please retry in 30-60s.")
    return {"models": inference_service.get_available_models()}


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    model_name: str = Form(default="resnet_50", description="Model to use for prediction. Use 'ab_test' for A/B testing."),
    top_k: int = Form(default=5, description="Number of top predictions to return"),
):
    """
    Predict crop disease from a leaf image.

    - Upload an image of a crop leaf
    - Optionally specify which model to use
    - Set model_name='ab_test' to enable A/B testing (random 50/50 split)
    - Returns predicted disease class with confidence scores
    """
    # Validate
    validate_image(image)

    # A/B test routing
    resolved_model, is_ab_test, ab_group = resolve_ab_test(model_name)

    if resolved_model not in MODEL_METADATA:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{resolved_model}'. Available: {list(MODEL_METADATA.keys())}",
        )

    if not inference_service:
        detail = f"Service failed to start: {_load_error}" if _load_error else "Models are still downloading/loading from GCS. Please retry in 30-60s."
        raise HTTPException(status_code=503, detail=detail)

    if inference_service and resolved_model not in inference_service.models:
        loaded = list(inference_service.models.keys())
        if not loaded:
            raise HTTPException(
                status_code=503,
                detail="Models are still downloading/loading from GCS. Please retry in 30-60s.",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' checkpoint not available on this server.",
        )

    # Read image
    img = await read_image(image)

    # Predict
    start_time = time.time()
    result = inference_service.predict(img, model_name=resolved_model, top_k=top_k)
    latency_ms = (time.time() - start_time) * 1000

    # Log
    ab_tag = f" | ab_test={ab_group}" if is_ab_test else ""
    logger.info(
        f"Prediction: {result['prediction']} ({result['confidence']:.2%}) "
        f"| model={resolved_model} | latency={latency_ms:.0f}ms{ab_tag}"
    )

    # Async W&B telemetry for A/B tests (does NOT block response)
    if is_ab_test:
        background_tasks.add_task(
            log_ab_telemetry,
            model_used=resolved_model,
            ab_group=ab_group,
            prediction=result["prediction"],
            confidence=result["confidence"],
            latency_ms=round(latency_ms, 1),
        )

    response = {
        **result,
        "latency_ms": round(latency_ms, 1),
    }
    if is_ab_test:
        response["ab_test"] = True
        response["ab_group"] = ab_group

    return response


@app.post("/explain")
@limiter.limit("10/minute")
async def explain(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    model_name: str = Form(default="resnet_50", description="Model to use. Use 'ab_test' for A/B testing."),
):
    """
    Predict + generate GradCAM heatmap overlay.

    Returns prediction results plus a base64-encoded heatmap image.
    Set model_name='ab_test' to enable A/B testing.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Validate
    validate_image(image)

    # A/B test routing
    resolved_model, is_ab_test, ab_group = resolve_ab_test(model_name)

    if resolved_model not in MODEL_METADATA:
        raise HTTPException(status_code=400, detail=f"Unknown model '{resolved_model}'.")

    if not inference_service:
        detail = f"Service failed to start: {_load_error}" if _load_error else "Models are still downloading/loading from GCS. Please retry in 30-60s."
        raise HTTPException(status_code=503, detail=detail)

    if inference_service and resolved_model not in inference_service.models:
        loaded = list(inference_service.models.keys())
        if not loaded:
            raise HTTPException(
                status_code=503,
                detail="Models are still downloading/loading from GCS. Please retry in 30-60s.",
            )
        raise HTTPException(status_code=404, detail=f"Model '{resolved_model}' not available.")

    # Read image
    img = await read_image(image)

    # Predict
    start_time = time.time()
    result = inference_service.predict(img, model_name=resolved_model, top_k=5)

    # GradCAM
    grayscale_cam = inference_service.get_gradcam(img, model_name=resolved_model)
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

    ab_tag = f" | ab_test={ab_group}" if is_ab_test else ""
    logger.info(
        f"Explain: {result['prediction']} ({result['confidence']:.2%}) "
        f"| model={resolved_model} | latency={latency_ms:.0f}ms{ab_tag}"
    )

    # Async W&B telemetry for A/B tests
    if is_ab_test:
        background_tasks.add_task(
            log_ab_telemetry,
            model_used=resolved_model,
            ab_group=ab_group,
            prediction=result["prediction"],
            confidence=result["confidence"],
            latency_ms=round(latency_ms, 1),
        )

    response = {
        **result,
        "heatmap_base64": heatmap_b64,
        "latency_ms": round(latency_ms, 1),
    }
    if is_ab_test:
        response["ab_test"] = True
        response["ab_group"] = ab_group

    return response


# ─── Error Handlers ───

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
