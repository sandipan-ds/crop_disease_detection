"""Tests for CropDiseaseDataset."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Test CSV is gitignored; skip tests if it is not present locally.
TEST_CSV = PROJECT_ROOT / "notebook" / "test.csv"
TRAIN_CSV = PROJECT_ROOT / "notebook" / "train.csv"
DATA_ROOT = PROJECT_ROOT


@pytest.mark.skipif(not TEST_CSV.exists(), reason=f"{TEST_CSV} not found (gitignored)")
class TestCropDiseaseDataset:
    def test_loads_without_error(self):
        from src.dataset import CropDiseaseDataset

        ds = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT))
        assert len(ds) > 0

    def test_returns_tensor_and_label(self):
        from src.dataset import CropDiseaseDataset

        ds = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT))
        image, label = ds[0]
        assert image.ndim == 3           # C x H x W
        assert isinstance(label, int)

    def test_get_labels_matches_length(self):
        from src.dataset import CropDiseaseDataset

        ds = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT))
        labels = ds.get_labels()
        assert len(labels) == len(ds)

    def test_get_class_names_non_empty(self):
        from src.dataset import CropDiseaseDataset

        ds = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT))
        names = ds.get_class_names()
        assert len(names) > 0
        assert len(names) == ds.num_classes

    def test_subsampling(self):
        from src.dataset import CropDiseaseDataset

        ds_full = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT))
        ds_small = CropDiseaseDataset(str(TEST_CSV), str(DATA_ROOT), sample_n=50)
        assert len(ds_small) <= 50
        assert ds_small.num_classes <= ds_full.num_classes
