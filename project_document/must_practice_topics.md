# Must-Practice Topics — Gemini Enterprise Agent Platform (Vertex AI)

> Based on what we have deployed and what we plan to deploy for the **Crop Disease Detection** project.

---

## 1. Already Used — Revise & Deepen

These are topics we've already touched in this project. Practice them until they feel second nature.

### 1.1 Service Accounts & IAM
- What a service account is and how it differs from a user account
- Creating a service account and downloading the JSON key
- Granting IAM roles (`Storage Object Admin`, `Vertex AI User`, `Service Account User`)
- **Principle of least privilege** — only give the minimum roles needed
- How `GOOGLE_APPLICATION_CREDENTIALS` environment variable works

> **What we did:** Created a service account, downloaded the key, granted `Storage Object Admin` on the bucket.

### 1.2 Cloud Storage (GCS)
- Creating buckets (naming rules, regions vs multi-regions)
- Uploading / downloading objects programmatically using `google-cloud-storage` SDK
- Bucket permissions vs project-level IAM
- `gsutil` CLI as an alternative to the Python SDK
- Storage classes (Standard, Nearline, Coldline) and when to use each
- Organizing data in GCS with prefixes (pseudo-folders)

> **What we did:** Created `crop-disease-detection-1` bucket in `asia` multi-region, uploaded ~61K images + CSVs + configs via Python script.

### 1.3 Environment & Secrets Management
- Using `.env` files with `python-dotenv`
- Why credentials must **never** be committed to git
- `.gitignore` best practices for GCP projects

> **What we did:** Moved all GCP config (project ID, bucket name, key path) into `.env`, added it to `.gitignore`.

---

## 2. Coming Next — Learn Before We Build

These are topics we will use in the next phases of the project.

### 2.1 Vertex AI Custom Training Jobs
- How custom training differs from AutoML
- Packaging a PyTorch training script for Vertex AI
- Pre-built containers vs custom Docker containers
- Specifying machine type and GPU (`n1-standard-8` + `NVIDIA_TESLA_T4`, etc.)
- Passing arguments to training scripts (hyperparameters)
- Reading data from GCS inside a training job
- Writing model artifacts back to GCS
- Monitoring training jobs in the console

> **Why:** We will submit our PyTorch crop disease model as a custom training job.

### 2.2 Docker & Containerization (for Vertex AI)
- Writing a `Dockerfile` for a PyTorch training environment
- Building and pushing images to **Google Container Registry (GCR)** or **Artifact Registry**
- Why Vertex AI needs containerized training code
- Using Google's pre-built PyTorch training containers to skip Docker entirely

> **Why:** Vertex AI runs training inside containers. You either use a pre-built one or build your own.

### 2.3 GPU Quota Management
- Checking GPU quota in a GCP region (`Quotas & System Limits` page)
- Requesting a quota increase (and how long it takes)
- Choosing the right GPU for your budget (`T4` for cost, `A100` for speed)
- Understanding `asia-south1` (Mumbai) GPU availability

> **Why:** New GCP projects often have 0 GPU quota. You must request it before submitting training jobs.

### 2.4 Vertex AI TensorBoard
- Integrating TensorBoard with Vertex AI training jobs
- Logging metrics (loss, accuracy, F1) during training
- Viewing TensorBoard experiments in the GCP console
- Comparing multiple training runs

> **Why:** We have `tensorboard` in our `requirements.txt` and will use it to monitor training.

---

## 3. Future / Post-Training — Worth Exploring

These topics become relevant after the model is trained.

### 3.1 Vertex AI Model Registry
- Uploading a trained model (`.pth` file) to the Model Registry
- Versioning models
- Associating metrics and metadata with model versions

### 3.2 Vertex AI Endpoints (Online Prediction)
- Deploying a model to an endpoint for real-time inference
- Choosing machine type for serving (CPU vs GPU)
- Sending prediction requests via REST API or Python SDK
- Autoscaling and traffic splitting between model versions
- Cost implications of always-on endpoints

### 3.3 Vertex AI Batch Prediction
- Running predictions on a large dataset without a live endpoint
- Input/output formats (JSONL, CSV, BigQuery)
- When to use batch vs online prediction

### 3.4 Vertex AI Pipelines (MLOps)
- What Kubeflow Pipelines / Vertex AI Pipelines are
- Automating the full workflow: data prep → training → evaluation → deployment
- Scheduling recurring training jobs
- CI/CD for ML models

### 3.5 Cost Management
- Understanding Vertex AI pricing (training hours, endpoint hours, storage)
- Setting up billing alerts
- Using preemptible/spot VMs for training to reduce cost
- Shutting down endpoints when not in use

---

## Quick Reference — GCP Console Links

| Resource | URL |
|---|---|
| IAM & Admin | [console.cloud.google.com/iam-admin](https://console.cloud.google.com/iam-admin) |
| Cloud Storage | [console.cloud.google.com/storage](https://console.cloud.google.com/storage) |
| Vertex AI | [console.cloud.google.com/vertex-ai](https://console.cloud.google.com/vertex-ai) |
| APIs & Services | [console.cloud.google.com/apis](https://console.cloud.google.com/apis) |
| Quotas | [console.cloud.google.com/iam-admin/quotas](https://console.cloud.google.com/iam-admin/quotas) |
| Billing | [console.cloud.google.com/billing](https://console.cloud.google.com/billing) |

---

## Key SDK Packages

| Package | Purpose |
|---|---|
| `google-cloud-storage` | Upload/download files to/from GCS |
| `google-cloud-aiplatform` | Submit training jobs, deploy models, manage endpoints |
| `python-dotenv` | Load `.env` files for credentials |
| `tensorboard` | Training metrics visualization |

---

*This document will be updated as the project progresses.*
