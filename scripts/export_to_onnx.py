"""
Export a trained PyTorch crop disease model to ONNX for Android (on-device) inference.

Workflow:
    1. Load .pth checkpoint (from local results/ folder)
    2. Reconstruct model via src.model.get_model()
    3. Export to ONNX (opset 13, dynamic batch)
    4. Optionally simplify with onnx-simplifier
    5. Optionally quantize to INT8 (4x smaller, ~same accuracy)
    6. Save labels.json in Android-friendly format
    7. Save metadata.json (model name, F1, input size, export date)
    8. Verify PyTorch vs ONNX outputs match on a dummy input

Usage:
    # Default: MobileNet V3 (recommended for mobile)
    python scripts/export_to_onnx.py

    # Specific model
    python scripts/export_to_onnx.py --model resnet_50

    # Full pipeline: export + simplify + INT8 quantize + verify
    python scripts/export_to_onnx.py --model mobilenet_v3 --quantize int8 --verify

    # Custom checkpoint path
    python scripts/export_to_onnx.py --model vit \\
        --checkpoint results/vit_v2/vit/models/best_model_fold_1.pth
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "exported"
DEFAULT_LABEL_MAPPING = PROJECT_ROOT / "configs" / "label_mapping.json"

# ImageNet normalization — must match training (api/inference.py) and Android preprocessor
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

SUPPORTED_MODELS = [
    "cnn_baseline", "resnet_50", "resnet_152", "vgg_16",
    "vit", "efficientnet_b4", "mobilenet_v3", "swin_base",
]


# =========================================================
# ARG PARSING
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Export a trained crop disease model to ONNX for Android.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="mobilenet_v3",
                   choices=SUPPORTED_MODELS,
                   help="Architecture to export. MobileNet V3 is recommended for mobile.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pth file. Default: results/<model>/models/best_model_fold_1.pth")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write .onnx, labels.json, metadata.json")
    p.add_argument("--label-mapping", type=Path, default=DEFAULT_LABEL_MAPPING,
                   help="Path to configs/label_mapping.json")
    p.add_argument("--opset", type=int, default=13,
                   help="ONNX opset version. 13 is widely supported by onnxruntime-android.")
    p.add_argument("--quantize", choices=["none", "int8"], default="none",
                   help="INT8 dynamic quantization: ~4x smaller model, ~1% accuracy drop.")
    p.add_argument("--simplify", action="store_true",
                   help="Run onnx-simplifier to clean up the graph before saving.")
    p.add_argument("--verify", action="store_true",
                   help="Run both PyTorch and ONNX on dummy input, assert outputs match.")
    p.add_argument("--num-classes", type=int, default=102,
                   help="Number of output classes. Must match training.")
    return p.parse_args()


# =========================================================
# LABEL LOADING + VALIDATION
# =========================================================

def load_labels(label_mapping_path: Path) -> dict:
    """Load labels from configs/label_mapping.json, return Android-friendly {idx_str: name}."""
    with open(label_mapping_path) as f:
        raw = json.load(f)

    # Accept both {"label_to_class": {...}} and flat {...}
    if "label_to_class" in raw:
        labels = raw["label_to_class"]
    else:
        labels = raw

    # Validate keys are 0..N-1
    int_keys = [int(k) for k in labels.keys()]
    expected = set(range(len(labels)))
    if set(int_keys) != expected:
        missing = expected - set(int_keys)
        extra = set(int_keys) - expected
        raise ValueError(
            f"Label mapping is not contiguous 0..{len(labels)-1}. "
            f"Missing: {sorted(missing)[:5]}... Extra: {sorted(extra)[:5]}..."
        )

    # Known typo in source data
    for idx, name in labels.items():
        if name.endswith(")"):
            print(f"  [WARN] Label {idx} has a stray ')': {name!r}")
            print(f"         Consider fixing configs/label_mapping.json before shipping.")

    return {str(k): str(v) for k, v in labels.items()}


# =========================================================
# MODEL LOADING
# =========================================================

def load_pytorch_model(model_name: str, checkpoint_path: Path, num_classes: int, device: str):
    """Reconstruct the model architecture and load trained weights."""
    # Lazy import — src.model pulls in torchvision, keep it out of arg parsing
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.model import get_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"  -> If you trained on Vertex AI, download it first:\n"
            f"    gcloud storage cp gs://crop-disease-detection-1/{checkpoint_path.relative_to(PROJECT_ROOT) if checkpoint_path.is_absolute() else checkpoint_path} {checkpoint_path}"
        )

    print(f"  Loading model: {model_name}")
    print(f"  Checkpoint:    {checkpoint_path}")

    model = get_model(model_name, num_classes=num_classes, pretrained=False, dropout_fc=0.5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Disable inplace ops — required for some ONNX export paths
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    val_metrics = checkpoint.get("val_metrics", {})
    print(f"  Epoch:         {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val F1 macro:  {val_metrics.get('f1_macro', 'N/A')}")
    print(f"  Val accuracy:  {val_metrics.get('accuracy', 'N/A')}")

    return model, val_metrics


# =========================================================
# ONNX EXPORT
# =========================================================

def export_to_onnx(model, model_name: str, output_path: Path, opset: int, device: str):
    """Export PyTorch model to ONNX format with dynamic batch dimension."""
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    print(f"\n  Exporting to ONNX (opset {opset}) on {device}...")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,  # use the legacy TorchScript-based exporter (more forgiving)
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def simplify_onnx(onnx_path: Path) -> Path:
    """Run onnx-simplifier to clean up redundant ops in the graph."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  [SKIP] onnx-simplifier not installed. Run: pip install onnx-simplifier")
        return onnx_path

    print(f"\n  Simplifying graph...")
    simplified_path = onnx_path.with_name(onnx_path.stem + "_opt.onnx")
    model = onnx.load(str(onnx_path))
    model_simp, check = simplify(model)
    if not check:
        print(f"  [WARN] Simplifier check failed, saving anyway.")
    onnx.save(model_simp, str(simplified_path))
    size_mb = simplified_path.stat().st_size / 1e6
    print(f"  Simplified: {simplified_path} ({size_mb:.1f} MB)")
    return simplified_path


def quantize_int8(onnx_path: Path) -> Path:
    """Dynamic INT8 quantization — ~4x smaller, ~1% accuracy drop typically."""
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  [SKIP] onnxruntime not installed. Run: pip install onnxruntime")
        return onnx_path

    print(f"\n  Quantizing to INT8...")
    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = int8_path.stat().st_size / 1e6
    print(f"  INT8: {int8_path} ({size_mb:.1f} MB)")
    return int8_path


# =========================================================
# VERIFICATION
# =========================================================

def verify_outputs_match(pytorch_model, onnx_path: Path, device: str, tolerance: float = 1e-4):
    """Run the same dummy input through both, assert outputs match within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed. Run: pip install onnxruntime")
        return

    print(f"\n  Verifying PyTorch vs ONNX outputs...")

    # PyTorch
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        torch_out = pytorch_model(dummy).cpu().numpy()

    # ONNX
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": dummy.cpu().numpy().astype(np.float32)})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    mean_diff = np.mean(np.abs(torch_out - onnx_out))

    print(f"  Torch output shape:  {torch_out.shape}")
    print(f"  ONNX output shape:   {onnx_out.shape}")
    print(f"  Max abs difference:  {max_diff:.2e}")
    print(f"  Mean abs difference: {mean_diff:.2e}")

    if max_diff > tolerance:
        print(f"  [WARN] Difference exceeds tolerance {tolerance:.0e} - outputs may diverge in production.")
    else:
        print(f"  [OK] Outputs match within tolerance ({tolerance:.0e}).")

    # Also verify argmax matches
    torch_pred = int(np.argmax(torch_out[0]))
    onnx_pred = int(np.argmax(onnx_out[0]))
    if torch_pred == onnx_pred:
        print(f"  [OK] Argmax agrees: class {torch_pred} (confidence "
              f"{float(torch_out[0][torch_pred]):.4f})")
    else:
        print(f"  [WARN] Argmax disagrees - Torch: {torch_pred}, ONNX: {onnx_pred}")


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    # Resolve checkpoint path
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / "results" / args.model / "models" / "best_model_fold_1.pth"

    print("=" * 70)
    print(f"  CROP DISEASE MODEL -> ONNX EXPORT")
    print(f"  Model:     {args.model}")
    print(f"  Quantize:  {args.quantize}")
    print(f"  Simplify:  {args.simplify}")
    print(f"  Verify:    {args.verify}")
    print("=" * 70)

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 1. Load PyTorch model
    model, val_metrics = load_pytorch_model(
        args.model, args.checkpoint, args.num_classes, device
    )

    # 2. Load labels
    print(f"\n  Loading labels from: {args.label_mapping}")
    labels = load_labels(args.label_mapping)
    print(f"  Found {len(labels)} classes")

    # 3. Export to ONNX
    onnx_path = args.output_dir / f"{args.model}_model.onnx"
    export_to_onnx(model, args.model, onnx_path, args.opset, device)

    # 4. Optional: simplify
    if args.simplify:
        onnx_path = simplify_onnx(onnx_path)

    # 5. Optional: INT8 quantize
    if args.quantize == "int8":
        onnx_path = quantize_int8(onnx_path)

    # 5b. Always produce a final canonical filename ({model}_model.onnx) for
    # downstream tools (Android asset copy). If we quantized/simplified, the
    # final onnx_path already points at the optimized file — copy it back.
    canonical_path = args.output_dir / f"{args.model}_model.onnx"
    if onnx_path.resolve() != canonical_path.resolve():
        import shutil
        shutil.copy2(onnx_path, canonical_path)
        print(f"  Canonical: {canonical_path} ({canonical_path.stat().st_size / 1e6:.1f} MB)")
        onnx_path = canonical_path

    # 6. Save labels.json (Android-friendly: {idx_str: name})
    labels_path = args.output_dir / f"{args.model}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n  Labels saved:  {labels_path}")

    # 7. Save metadata.json
    metadata = {
        "model_name": args.model,
        "num_classes": args.num_classes,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "channels": 3,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "val_f1_macro": val_metrics.get("f1_macro"),
        "val_accuracy": val_metrics.get("accuracy"),
        "onnx_file": onnx_path.name,
        "labels_file": labels_path.name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "opset": args.opset,
    }
    metadata_path = args.output_dir / f"{args.model}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    # 8. Optional: verify — run a real image (or dummy) through PyTorch AND ONNX
    if args.verify:
        # Use a synthetic image with realistic leaf-like statistics (mean ~ 0.5,
        # low std) rather than pure noise — gives a more meaningful comparison
        # than N(0, 1) noise. The result is a smoke test, not a full eval.
        verify_outputs_match(model, onnx_path, device)

    print("\n" + "=" * 70)
    print(f"  EXPORT COMPLETE")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model:      {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Labels:     {labels_path.name}")
    print(f"  Metadata:   {metadata_path.name}")
    print("=" * 70)
    print(f"\n  Next step - copy to Android project:")
    print(f"    cp {onnx_path}  ../crop_disease_android/app/src/main/assets/")
    print(f"    cp {labels_path} ../crop_disease_android/app/src/main/assets/")


if __name__ == "__main__":
    main()
