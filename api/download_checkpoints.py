"""
Download model checkpoints from GCS at container startup.

Uses signed URL approach to avoid ADC auth hangs on Cloud Run.
Falls back to direct client download if signing is unavailable.
"""

import os
import sys
import traceback
import urllib.request
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage
from google.api_core.exceptions import NotFound

GCS_BUCKET = os.environ.get("GCS_BUCKET", "crop-disease-detection-1")
MODELS_TO_DOWNLOAD = os.environ.get(
    "MODELS_TO_DOWNLOAD",
    "resnet_50,mobilenet_v3,resnet_152,vit,swin_base"
).split(",")

LOCAL_RESULTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "results"))
MAX_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 300


def _download_via_signed_url(bucket, blob, local_path: Path, model_name: str) -> str:
    """Download using a signed URL — bypasses auth handshake per-request."""
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=DOWNLOAD_TIMEOUT_SEC + 60,
        method="GET",
    )
    print(f" [{model_name}] Signed URL generated, downloading via urllib...", flush=True)
    tmp_path = local_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(signed_url, str(tmp_path))
        tmp_path.rename(local_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    size_mb = local_path.stat().st_size / 1e6
    print(f" [{model_name}] Download complete ({size_mb:.1f} MB)", flush=True)
    return f"{model_name}: downloaded ({size_mb:.1f} MB)"


def _download_via_client(bucket, blob, local_path: Path, model_name: str) -> str:
    """Fallback: download via google-cloud-storage client using download_as_bytes.

    download_to_filename uses resumable media downloads which can hang on Cloud Run.
    download_as_bytes does a single HTTP GET and loads into memory, then we write
    the file ourselves. Model files are ~100-200MB which fits in Cloud Run's 4Gi RAM.
    """
    print(f" [{model_name}] Downloading via storage client (download_as_bytes, timeout={DOWNLOAD_TIMEOUT_SEC}s)...", flush=True)
    try:
        data = blob.download_as_bytes(timeout=DOWNLOAD_TIMEOUT_SEC)
        print(f" [{model_name}] Received {len(data) / 1e6:.1f} MB, writing to disk...", flush=True)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        del data
    except Exception:
        if local_path.exists():
            local_path.unlink()
        raise
    size_mb = local_path.stat().st_size / 1e6
    print(f" [{model_name}] Download complete ({size_mb:.1f} MB)", flush=True)
    return f"{model_name}: downloaded ({size_mb:.1f} MB)"


def _download_one(bucket, model_name: str) -> str:
    """Download a single model checkpoint. Returns status message."""
    model_name = model_name.strip()
    if not model_name:
        return "skipped (empty name)"

    gcs_path = f"results/{model_name}/models/best_model_fold_1.pth"
    local_path = LOCAL_RESULTS_DIR / model_name / "models" / "best_model_fold_1.pth"

    if local_path.exists():
        size_mb = local_path.stat().st_size / 1e6
        print(f" [{model_name}] Already exists ({size_mb:.1f} MB)", flush=True)
        return f"{model_name}: already exists ({size_mb:.1f} MB)"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f" [{model_name}] Starting download from gs://{bucket.name}/{gcs_path} -> {local_path}", flush=True)
    try:
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            print(f" [{model_name}] Blob does NOT exist in GCS", flush=True)
            return f"{model_name}: NOT FOUND in GCS"

        print(f" [{model_name}] Blob exists, size={blob.size}", flush=True)

        try:
            return _download_via_signed_url(bucket, blob, local_path, model_name)
        except Exception as sign_err:
            print(f" [{model_name}] Signed URL failed ({sign_err}), falling back to client download...", flush=True)
            return _download_via_client(bucket, blob, local_path, model_name)

    except NotFound:
        print(f" [{model_name}] NOT FOUND in GCS", flush=True)
        return f"{model_name}: NOT FOUND in GCS"
    except Exception as e:
        print(f" [{model_name}] ERROR: {e}", flush=True)
        traceback.print_exc()
        if local_path.exists():
            local_path.unlink()
        tmp_path = local_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        return f"{model_name}: ERROR {e}"


def download_checkpoints():
    """Download model checkpoints from GCS bucket sequentially."""
    models = [m.strip() for m in MODELS_TO_DOWNLOAD if m.strip()]
    print(f"[download_checkpoints] Starting sequential download. Bucket={GCS_BUCKET}, Models={models}", flush=True)

    print("[download_checkpoints] Creating storage.Client()...", flush=True)
    try:
        client = storage.Client()
        print("[download_checkpoints] Client created.", flush=True)

        creds = client._credentials
        print(f"[download_checkpoints] Credential type: {type(creds).__name__}", flush=True)
        print(f"[download_checkpoints] Project: {client.project}", flush=True)

        bucket = client.bucket(GCS_BUCKET)
        print(f"[download_checkpoints] Bucket reference: {bucket.name}", flush=True)

        try:
            bucket.reload()
            print(f"[download_checkpoints] Bucket exists: True, location={bucket.location}", flush=True)
        except NotFound:
            print(f"[download_checkpoints] Bucket NOT FOUND: {GCS_BUCKET}", flush=True)
            return
        except Exception as e:
            print(f"[download_checkpoints] Bucket access check failed: {e}", flush=True)
            print(f"[download_checkpoints] WARNING: Service account may lack storage.objectViewer on bucket!", flush=True)

    except Exception as e:
        print(f"[download_checkpoints] FAILED to create GCS client: {e}", flush=True)
        traceback.print_exc()
        return

    downloaded = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_one, bucket, name): name for name in models}
        for future in as_completed(futures, timeout=DOWNLOAD_TIMEOUT_SEC * len(models)):
            result = future.result()
            print(f"[download_checkpoints] {result}", flush=True)
            if "downloaded" in result:
                downloaded += 1

    print(f"[download_checkpoints] Done. Downloaded {downloaded}/{len(models)} models.", flush=True)


if __name__ == "__main__":
    download_checkpoints()
