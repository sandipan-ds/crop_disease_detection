# 🌿 Crop Disease Detection — Computer Vision for Smart Agriculture

## Business Objective

### Executive Problem Statement

Global agriculture faces massive challenges as plant diseases and pests threaten food security and crop yields, with diseases contributing to massive yield losses annually. Traditional manual monitoring of crop health relies on farmers visually inspecting fields — a process that is labor-intensive, prone to human error, and often catches outbreaks too late. Waiting too long to take action leads to devastating crop loss or the indiscriminate, environmentally harmful overuse of chemical pesticides.

### Project Goal

The objective of this project is to develop a **deep learning-based image classifier** utilizing **Computer Vision**. The model must analyze images of plant leaves to detect early signs of disease, enabling timely, targeted interventions and supporting precision agriculture objectives.

### Strategic Vision

Democratize expert agricultural knowledge by giving local farmers instant, smartphone-accessible diagnostic capabilities. Early and accurate detection mitigates damage and ensures sustainable food production.

### Key Performance Indicators

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness of predictions across all 38 disease classes |
| **Precision** | Minimize false positives — avoid unnecessary pesticide application |
| **Recall** ⭐ | **Most critical** — a false negative (missed disease) allows infection to spread across the field |
| **F1-Score** | Harmonic mean balancing Precision and Recall |

### User Personas

| Persona | Primary Need | System Interaction |
|---|---|---|
| **Local Farmer** | Fast, reliable diagnosis without expert botanical knowledge | Captures a photo of a suspicious leaf via mobile → receives instant AI diagnosis and treatment recommendation |
| **Agronomist** | Scalable monitoring of large geographic farming zones | Uses aggregated model predictions to track disease outbreak patterns across regions |
| **ML Engineer** | High-quality image preprocessing and robust model generalization | Augments limited image datasets and tunes CNNs to handle varying field lighting conditions |

---

## Project Structure

```
crop_disease_detection/
│
├── business_objective/           # PDF with project brief (gitignored)
│   └── Project 2.pdf
│
├── configs/                      # YAML configuration files
│   └── training_config.yaml      # Hyperparams, Vertex AI settings, augmentation
│
├── data/                         # Raw & processed datasets (gitignored)
│   ├── original/                 # Unmodified PlantVillage download
│   └── processed/                # Resized, split into train/val/test
│
├── models/
│   └── saved/                    # Trained model weights (.pth files)
│
├── notebook/
│   └── crop_disease_detection.ipynb   # EDA, prototyping, experimentation
│
├── reports/                      # Generated evaluation outputs
│   └── .gitkeep                  # Confusion matrices, loss curves, reports
│
├── scripts/                      # Standalone automation scripts
│   ├── upload_to_gcs.py          # Push local data → GCS bucket
│   └── submit_vertex_job.py      # Launch Vertex AI training job
│
├── src/                          # Main source package
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset class, data loaders
│   │   ├── preprocessing.py      # Resize, normalize, color-space conversion
│   │   └── augmentation.py       # Transforms (flips, rotations, color jitter)
│   │
│   ├── models/
│   │   ├── custom_cnn.py         # Custom CNN architecture from scratch
│   │   └── transfer_learning.py  # ResNet50 / EfficientNet / MobileNetV2
│   │
│   ├── training/
│   │   ├── trainer.py            # Training loop, early stopping, checkpoints
│   │   └── vertex_ai.py          # Vertex AI job submission & configuration
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, Precision, Recall, F1, confusion matrix
│   │   └── visualize.py          # Loss curves, Grad-CAM, prediction grids
│   │
│   ├── inference/
│   │   └── predict.py            # Load model → preprocess image → predict disease
│   │
│   └── utils/
│       ├── config.py             # Load YAML config, default hyperparameters
│       └── gcs_utils.py          # GCS upload/download helpers
│
├── .gitignore
├── Dockerfile                    # Vertex AI custom training container
├── README.md
└── requirements.txt              # All Python dependencies
```

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Deep Learning** | PyTorch, torchvision | Specialized tensor operations, GPU acceleration, rich pre-trained model zoo |
| **Image Processing** | OpenCV, Pillow, Albumentations | Industry-standard manipulation, resizing, and fast augmentation pipelines |
| **Data Science** | NumPy, Pandas, scikit-learn | Array ops, tabular metrics logging, train/test splitting, classification reports |
| **Visualization** | Matplotlib, Seaborn | Training curves, confusion matrix heatmaps, sample prediction grids |
| **Cloud Training** | Google Vertex AI | On-demand GPU (T4/V100) for training large image datasets at scale |
| **Storage** | Google Cloud Storage | Store PlantVillage dataset and model weights in GCS buckets |
| **Experiment Tracking** | TensorBoard | Real-time training metric visualization, integrates with Vertex AI |
| **Dataset** | PlantVillage | Open-source repository of ~55,000 labeled images of healthy and diseased crop leaves |

---

## Four-Week Engineering Roadmap

### Week 1 — Image Acquisition, EDA & Preprocessing
- Download a subset of the PlantVillage dataset
- Exploratory Data Analysis: plot sample images, analyze class distribution, check for imbalances
- Build an automated preprocessing pipeline (resize to 224×224, normalize pixel arrays, train/val/test split)

### Week 2 — Custom CNN Architecture & Baseline Training
- Construct a custom CNN from scratch (Conv2D → BatchNorm → ReLU → MaxPool blocks)
- Configure categorical cross-entropy loss and train the baseline model
- Monitor training/validation loss curves; apply Dropout and Early Stopping to mitigate overfitting

### Week 3 — Transfer Learning & Hyperparameter Optimization
- Import pre-trained architectures (ResNet50 / MobileNetV2 / EfficientNet-B0)
- Freeze base layers, fine-tune the classification head for crop diseases
- Experiment with hyperparameter tuning (learning rates, batch sizes) — target accuracy **>90%**

### Week 4 — Evaluation, Inference & Deployment
- Generate Confusion Matrix and per-class Classification Report
- Save final model weights; write an inference script (image path → predicted disease + confidence)
- Full repository documentation with commit history showing iterative architecture improvements

---

## Getting Started

### Prerequisites
- Python 3.10+
- A GCP project with Vertex AI API and Cloud Storage API enabled
- `gcloud` CLI authenticated
- Docker (for Vertex AI container builds)

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd crop_disease_detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training on Vertex AI
```bash
# 1. Upload dataset to GCS
python scripts/upload_to_gcs.py --bucket gs://your-bucket --source data/processed

# 2. Build and push training container
docker build -t crop-disease-training .
docker tag crop-disease-training us-docker.pkg.dev/YOUR_PROJECT/repo/crop-disease-training
docker push us-docker.pkg.dev/YOUR_PROJECT/repo/crop-disease-training

# 3. Submit training job
python scripts/submit_vertex_job.py --config configs/training_config.yaml
```

---

## License

This project is for educational and research purposes.
