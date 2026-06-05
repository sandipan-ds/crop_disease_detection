# =========================================================
# Makefile — Crop Disease Detection
#
# Provides convenient targets for data prep, validation,
# training, testing, and deployment workflows.
#
# Usage:
#   make prepare    — run data preparation pipeline
#   make validate   — validate dataset manifests
#   make train      — submit training job to Vertex AI
#   make test       — run pytest suite
#   make deploy     — build and deploy backend to Cloud Run
#   make dvc-push   — push DVC-tracked data to GCS remote
#   make dvc-pull   — pull DVC-tracked data from GCS remote
# =========================================================

PYTHON := python
DVC := dvc

.PHONY: help prepare validate train export-onnx test deploy dvc-push dvc-pull lint format clean

help:
	@echo "Available targets:"
	@echo "  prepare     — run data preparation (rename images, generate CSVs)"
	@echo "  validate    — validate CSV manifests and image paths"
	@echo "  train       — submit a Vertex AI training job"
	@echo "  export-onnx — export trained model to ONNX for Android (default: mobilenet_v3)"
	@echo "  test        — run backend test suite with pytest"
	@echo "  deploy      — build Docker image and deploy to Cloud Run"
	@echo "  dvc-push    — push DVC-tracked datasets to GCS remote"
	@echo "  dvc-pull    — pull DVC-tracked datasets from GCS remote"
	@echo "  lint        — run ruff / flake8 checks"
	@echo "  format      — auto-format Python code with black / ruff"
	@echo "  clean       — remove Python caches and temp files"

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

prepare:
	$(PYTHON) scripts/prepare_data.py \
		--train-dir data/processed/combined_train \
		--test-dir data/processed/combined_test \
		--output-dir data/csv

validate:
	$(PYTHON) scripts/validate_data.py \
		--train-csv data/csv/train.csv \
		--test-csv data/csv/test.csv \
		--label-mapping configs/label_mapping.json

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

train:
	@echo "Submitting training job to Vertex AI..."
	$(PYTHON) scripts/submit_vertex_job.py --config configs/training_config.yaml

export-onnx:
	@echo "Exporting trained model to ONNX for Android (MobileNet V3, INT8, verified)..."
	$(PYTHON) scripts/export_to_onnx.py --model mobilenet_v3 --quantize int8 --simplify --verify

# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ---------------------------------------------------------
# Deployment
# ---------------------------------------------------------

deploy:
	@echo "Building Docker image..."
	gcloud builds submit \
		--tag us-central1-docker.pkg.dev/crop-disease-detection-496608/crop-disease-api/crop-disease-api:latest \
		--dockerfile api/Dockerfile \
		--timeout=1800
	@echo "Deploying to Cloud Run..."
	gcloud run deploy crop-disease-api \
		--image us-central1-docker.pkg.dev/crop-disease-detection-496608/crop-disease-api/crop-disease-api:latest \
		--region us-central1

# ---------------------------------------------------------
# DVC
# ---------------------------------------------------------

dvc-push:
	$(DVC) push

dvc-pull:
	$(DVC) pull

# ---------------------------------------------------------
# Code Quality
# ---------------------------------------------------------

lint:
	ruff check scripts/ src/ api/ tests/

format:
	ruff format scripts/ src/ api/ tests/

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
