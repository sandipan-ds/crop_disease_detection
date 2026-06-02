# 🌿 Crop Disease Detection — End-to-End ML Platform

[![Frontend](https://img.shields.io/badge/Frontend-Live-green)](https://crop-disease-detection-30ba1.web.app)
[![Backend](https://img.shields.io/badge/API-Live-blue)](https://crop-disease-api-1049249498032.us-central1.run.app/docs)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blueviolet)](.github/workflows/)
[![DVC](https://img.shields.io/badge/DVC-GCS%20Remote-orange)](data/dvc/)

> **Live Demo:** [https://crop-disease-detection-30ba1.web.app](https://crop-disease-detection-30ba1.web.app)
>
> **API Docs:** [https://crop-disease-api-1049249498032.us-central1.run.app/docs](https://crop-disease-api-1049249498032.us-central1.run.app/docs)

## Business Objective

### Executive Problem Statement

Global agriculture faces massive challenges as plant diseases and pests threaten food security and crop yields. Traditional manual monitoring relies on farmers visually inspecting fields — a process that is labor-intensive, prone to human error, and often catches outbreaks too late.

### Project Goal

Develop a **deep learning-based image classifier** that analyzes plant leaf images to detect early signs of disease across **102 crop disease classes** and **20 plant species**, enabling timely, targeted interventions.

### Strategic Vision

Democratize expert agricultural knowledge by giving farmers instant, smartphone-accessible diagnostic capabilities with **explainable AI** (GradCAM heatmaps) and **multi-model comparison**.

### Key Performance Indicators

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness across all 102 disease classes |
| **Precision** | Minimize false positives — avoid unnecessary pesticide application |
| **Recall** ⭐ | **Most critical** — a false negative allows infection to spread |
| **F1-Score (macro)** | Balances Precision and Recall; treats all classes equally |

---

## Architecture

```
User Browser
    |
    v
Firebase Hosting (React Frontend)
    |  https://crop-disease-detection-30ba1.web.app
    v
Cloud Run (FastAPI + PyTorch Backend)
    |  https://crop-disease-api-1049249498032.us-central1.run.app
    v
Google Cloud Storage (Model Checkpoints)
    |  gs://crop-disease-detection-1/
    v
Trained Models (.pth files)
```

## Project Structure

```
crop_disease_detection/
│
├── api/                          # FastAPI backend
│   ├── main.py                   # FastAPI app with /health, /models, /predict, /explain
│   ├── inference.py              # Model loading, prediction, GradCAM
│   ├── Dockerfile                # Cloud Run container
│   ├── requirements.txt          # API dependencies
│   └── run_local.py              # Local dev server (port 8000)
│
├── frontend/                     # React web app
│   ├── src/
│   │   ├── App.jsx               # Main app with tabs, state, API polling
│   │   ├── api.js                # Axios client (Cloud Run URL)
│   │   └── components/
│   │       ├── ImageUpload.jsx   # Drag-and-drop upload
│   │       ├── ModelSelector.jsx # Model dropdown with F1 scores
│   │       ├── ResultsPanel.jsx  # Diagnosis + GradCAM display
│   │       ├── ComparePanel.jsx  # Side-by-side GradCAM comparison
│   │       └── Header.jsx        # App header
│   ├── vite.config.js            # Vite build config
│   └── package.json
│
├── configs/
│   └── label_mapping.json        # 102-class label mapping
│
├── data/
│   ├── dvc/                      # DVC metadata vault (Git-tracked .dvc recipes)
│   ├── processed/                # Renamed + organized images (DVC-tracked)
│   │   ├── combined_train/
│   │   └── combined_test/
│   ├── csv/                      # Generated manifests (DVC-tracked)
│   │   ├── train.csv
│   │   └── test.csv
│   └── original/                 # Raw source datasets (Git-ignored, not DVC-tracked)
│
├── notebook/
│   └── crop_disease_detection.ipynb  # EDA, model evaluation, PAC analysis
│
├── scripts/
│   ├── prepare_data.py           # Automated data prep (rename → CSV → label map)
│   ├── validate_data.py          # Lightweight data validation checks
│   ├── upload_to_gcs.py          # Upload datasets to GCS bucket
│   ├── submit_vertex_job.py      # Launch Vertex AI training
│   └── vertex_ai_training.py     # Cloud training script
│
├── src/
│   ├── dataset.py                # PyTorch Dataset from CSV manifests
│   ├── model.py                  # 8 model architectures (CNN, ResNet, ViT, etc.)
│   ├── augmentations.py          # GPU-accelerated torchvision v2 transforms
│   └── training/
│       └── trainer.py            # Training loop with checkpointing
│
├── configs/
│   └── label_mapping.json        # 102-class label mapping
│
├── tests/                        # Pytest suite for FastAPI backend
│
├── project_document/
│   └── documentation/            # Modular docs (INDEX.md + per-section guides)
│
├── .github/workflows/            # CI/CD: lint, test, build, deploy
│
├── dvc.yaml                      # DVC pipeline: data preparation stage
├── Makefile                      # Convenient targets: make prepare, train, deploy
├── .firebaserc                   # Firebase project alias
├── firebase.json                 # Firebase Hosting config
├── Dockerfile                    # Vertex AI training container
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Deep Learning** | PyTorch, torchvision | GPU acceleration, rich pre-trained model zoo |
| **Explainability** | pytorch-grad-cam | GradCAM heatmaps for model interpretability |
| **Backend** | FastAPI, Uvicorn | High-performance async API with auto-generated docs |
| **Frontend** | React 19, Vite 8, TailwindCSS 4 | Modern, fast, responsive UI |
| **Image Processing** | OpenCV, Pillow | Resize, normalize, color-space conversion |
| **Data Science** | NumPy, Pandas, scikit-learn | Metrics, splitting, classification reports |
| **Visualization** | Matplotlib, Seaborn | Training curves, confusion matrices |
| **Cloud Training** | Google Vertex AI | On-demand GPU (T4/V100/A100) |
| **Cloud Backend** | Google Cloud Run | Serverless, auto-scales to zero |
| **Cloud Frontend** | Firebase Hosting | Free static hosting with CDN |
| **Storage** | Google Cloud Storage | Dataset and model checkpoints |
| **Experiment Tracking** | TensorBoard | Real-time metric visualization |
| **Data Versioning** | DVC + GCS | Reproducible datasets and model checkpoints |
| **CI/CD** | GitHub Actions | Automated test, build, deploy on push |
| **Dataset** | PlantVillage + plant_dataset_2 | ~61,000 labeled images, 102 classes |

---

## Model Performance

All models trained with ImageNet pretrained weights + custom classification heads.

| Model | Type | Best Val F1 (macro) |
|---|---|---|
| CNN Baseline | Custom CNN | 0.8309 |
| ResNet-50 | Transfer Learning | 0.9360 |
| EfficientNet-B4 | Transfer Learning | 0.8942 |
| VGG-16 | Transfer Learning | 0.8708 |
| ViT (B/16) | Transfer Learning | 0.9177 |
| **ResNet-152** | Transfer Learning | **0.9519** ⭐ |
| MobileNet V3 | Transfer Learning | 0.9231 |
| Swin-Base | Transfer Learning | 0.9271 |

**Dataset:** 102 classes | 42,006 train | 19,167 test | 20 plant species

## Features

- **Multi-model inference** — select from 8 trained models
- **GradCAM explainability** — visual heatmaps showing model attention
- **Side-by-side comparison** — compare GradCAM across 2-3 models simultaneously
- **Cold-start handling** — frontend polls backend and shows loading state
- **Scale-to-zero backend** — Cloud Run costs minimized when idle
- **Rate limiting** — 30/min predict, 10/min explain
- **DVC data versioning** — datasets and models tracked with GCS remote
- **CI/CD pipeline** — GitHub Actions for automated test, build, and deploy

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if models are loaded |
| `/models` | GET | List available models with F1 scores |
| `/predict` | POST | Upload image + model → prediction + confidence |
| `/explain` | POST | Upload image + model → GradCAM heatmap (base64) |

See full API docs at: [https://crop-disease-api-1049249498032.us-central1.run.app/docs](https://crop-disease-api-1049249498032.us-central1.run.app/docs)

---

## Quick Start

### Use the Live App (No Setup)

1. Open [https://crop-disease-detection-30ba1.web.app](https://crop-disease-detection-30ba1.web.app)
2. Upload a leaf image
3. Select a model from the dropdown
4. Click **Predict** → get diagnosis + confidence
5. Click **Explain GradCAM** → see heatmap
6. Switch to **Compare Models** → compare 2-3 models side-by-side

> **Note:** First prediction after idle may take 30-60s (Cloud Run cold start).

### Local Development

#### Prerequisites
- Python 3.11+
- Node.js 18+
- `gcloud` CLI (for deployment)

#### Backend (FastAPI)
```cmd
cd api
python -m run_local
```
Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Test endpoints:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
curl -X POST http://localhost:8000/predict -F "image=@test.jpg" -F "model_name=resnet_152"
curl -X POST http://localhost:8000/explain -F "image=@test.jpg" -F "model_name=resnet_152"
```

#### Frontend (React)
```cmd
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173/`).

#### Deploy Frontend Updates
```cmd
cd frontend
npm run build
cd ..
firebase deploy
```

#### Deploy Backend Updates
```cmd
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/crop-disease-detection-496608/crop-disease-api/crop-disease-api:latest \
    --dockerfile api/Dockerfile \
    --timeout=1800

gcloud run deploy crop-disease-api \
    --image us-central1-docker.pkg.dev/crop-disease-detection-496608/crop-disease-api/crop-disease-api:latest \
    --region us-central1
```

### Data Preparation (Automated)

```cmd
# One-command data prep: rename images → generate CSVs → build label mapping
make prepare

# Validate manifests and image paths
make validate
```

### Training on Vertex AI

```bash
# Submit training job
make train

# Or directly:
python scripts/submit_vertex_job.py --config configs/training_config.yaml
```

---

## License

This project is for educational and research purposes.
