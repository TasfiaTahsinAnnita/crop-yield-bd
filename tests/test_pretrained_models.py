"""Checks that the API loads saved estimators instead of fitting new ones."""
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np

from python_backend import main


class FittedEstimator:
    def __init__(self, feature_names=None):
        self.feature_names_in_ = np.array(feature_names) if feature_names else None
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("Pretrained estimators must not be fitted at startup")

    def predict(self, _features):
        return np.array([4.0])


class SavedExecutor:
    dtype_byte_size = 2

    def to(self, devices, _dtype, _dtype_byte_size):
        self.devices = devices


class PretrainedModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.previous_models = main.models.copy()
        self.previous_metrics = main.metrics.copy()
        self.previous_statuses = {name: status.copy() for name, status in main.statuses.items()}
        self.previous_imputer = main.imputer

    def tearDown(self):
        main.models.clear()
        main.models.update(self.previous_models)
        main.metrics.clear()
        main.metrics.update(self.previous_metrics)
        main.statuses.clear()
        main.statuses.update(self.previous_statuses)
        main.imputer = self.previous_imputer

    def test_loads_saved_estimators_without_fitting(self):
        metadata = {
            "target": main.TARGET,
            "feature_columns": main.FEATURES,
            "numeric_columns": main.NUMERIC,
            "categorical_columns": main.CATEGORICAL,
            "training_rows": 1000,
            "dataset": "bangladesh_crop_yield_synthetic_1000.csv",
            "training_note": "trained on all rows",
        }
        saved_imputer = type("SavedImputer", (), {"feature_names_in_": np.array(main.NUMERIC)})()
        tabfm = FittedEstimator()
        tabpfn = FittedEstimator(main.FEATURES)
        tabpfn.executor_ = SavedExecutor()
        tabpfn.forced_inference_dtype_ = None
        metadata_file = mock_open(read_data=json.dumps(metadata))

        with (
            patch.object(Path, "open", metadata_file),
            patch.object(
                main,
                "_load_joblib_artifact",
                side_effect=[saved_imputer, tabfm, tabpfn],
            ),
            patch.object(main.torch.cuda, "is_available", return_value=False),
        ):
            loaded_metadata = main._load_pretrained_models()

        self.assertEqual(loaded_metadata, metadata)
        self.assertIs(main.models["TabFM"], tabfm)
        self.assertIs(main.models["TabPFN"], tabpfn)
        self.assertTrue(all(status["ready"] for status in main.statuses.values()))
        self.assertEqual(main.metrics, {})
        self.assertEqual((tabfm.fit_calls, tabpfn.fit_calls), (0, 0))
        self.assertEqual(tabpfn.device, "cpu")
        self.assertEqual(tabpfn.devices_, (main.torch.device("cpu"),))
        self.assertEqual(tabpfn.executor_.devices, tabpfn.devices_)


if __name__ == "__main__":
    unittest.main()
