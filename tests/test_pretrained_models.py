"""Checks that the API loads saved estimators instead of fitting new ones."""
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
from pydantic import ValidationError

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


class SavedImputer:
    def transform(self, values):
        return values.to_numpy()


class PretrainedModelLoadingTests(unittest.TestCase):
    @staticmethod
    def prediction_payload(**overrides):
        values = {
            "region": "Rangpur",
            "year": 2020,
            "crop": "Maize",
            "season": "Rabi",
            "rainfall_mm_season": 100,
            "avg_temperature_c": 25,
            "relative_humidity_pct": 60,
            "soil_ph": 6,
            "soil_moisture_m3_m3": 0.3,
            "solar_radiation_mj_m2_day": 15,
            "ndvi": 0.5,
            "fertilizer_kg_ha": 100,
            "irrigation_used_0_no_1_yes": 1,
            "previous_season_yield_tonnes_ha": 4,
        }
        values.update(overrides)
        return main.PredictionInput(**values)

    def setUp(self):
        self.previous_models = main.models.copy()
        self.previous_metrics = main.metrics.copy()
        self.previous_statuses = {name: status.copy() for name, status in main.statuses.items()}
        self.previous_imputer = main.imputer
        self.previous_feature_ranges = main.feature_ranges.copy()
        self.previous_combinations = main.supported_crop_combinations.copy()

    def tearDown(self):
        main.models.clear()
        main.models.update(self.previous_models)
        main.metrics.clear()
        main.metrics.update(self.previous_metrics)
        main.statuses.clear()
        main.statuses.update(self.previous_statuses)
        main.imputer = self.previous_imputer
        main.feature_ranges = self.previous_feature_ranges
        main.supported_crop_combinations = self.previous_combinations

    def test_loads_both_pretrained_models_at_startup(self):
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
            patch.object(Path, "is_file", return_value=True),
        ):
            loaded_metadata = main._load_pretrained_models()

        self.assertEqual(loaded_metadata, metadata)
        self.assertIs(main.models["TabFM"], tabfm)
        self.assertIs(main.models["TabPFN"], tabpfn)
        self.assertTrue(main.statuses["TabFM"]["ready"])
        self.assertTrue(main.statuses["TabFM"]["loaded"])
        self.assertTrue(main.statuses["TabPFN"]["ready"])
        self.assertTrue(main.statuses["TabPFN"]["loaded"])
        self.assertEqual(main.metrics, {})
        self.assertEqual((tabfm.fit_calls, tabpfn.fit_calls), (0, 0))
        self.assertEqual(tabpfn.device, "cpu")
        self.assertEqual(tabpfn.devices_, (main.torch.device("cpu"),))
        self.assertEqual(tabpfn.executor_.devices, tabpfn.devices_)

    def test_prediction_input_rejects_model_selection(self):
        with self.assertRaises(ValidationError):
            main.PredictionInput(
                model="TabPFN",
                region="Rangpur",
                year=2020,
                crop="Maize",
                season="Rabi",
                rainfall_mm_season=100,
                avg_temperature_c=25,
                relative_humidity_pct=60,
                soil_ph=6,
                soil_moisture_m3_m3=0.3,
                solar_radiation_mj_m2_day=15,
                ndvi=0.5,
                fertilizer_kg_ha=100,
                irrigation_used_0_no_1_yes=1,
                previous_season_yield_tonnes_ha=4,
            )

    def test_health_is_degraded_when_either_model_is_unavailable(self):
        statuses = {
            "TabFM": {"ready": True, "loaded": True},
            "TabPFN": {"ready": False, "loaded": False},
        }
        with (
            patch.object(main, "metadata", {"dataset_loaded": True}),
            patch.object(main, "imputer", SavedImputer()),
            patch.object(main, "statuses", statuses),
        ):
            result = main.health()

        self.assertEqual(result["service"], "degraded")

    def test_prediction_always_runs_both_models_concurrently(self):
        payload = self.prediction_payload()
        categories = {
            "region": ["Rangpur"],
            "year": [2020],
            "crop": ["Maize"],
            "season": ["Rabi"],
        }
        ranges = {column: (-100000, 100000) for column in main.NUMERIC}
        barrier = threading.Barrier(2)

        class ConcurrentEstimator(FittedEstimator):
            def predict(self, features):
                barrier.wait(timeout=5)
                return super().predict(features)

        estimators = {"TabPFN": ConcurrentEstimator(), "TabFM": ConcurrentEstimator()}

        with (
            patch.object(main, "metadata", {"dataset_loaded": True, "categories": categories, "synthetic": True}),
            patch.object(main, "imputer", SavedImputer()),
            patch.object(main, "feature_ranges", ranges),
            patch.object(main, "supported_crop_combinations", {("Rangpur", "Maize", "Rabi")}),
            patch.object(main, "_load_model", side_effect=estimators.__getitem__),
        ):
            result = main.predict(payload)

        self.assertEqual(set(result["results"]), {"TabPFN", "TabFM"})
        self.assertTrue(all(value["status"] == "ok" for value in result["results"].values()))

    def test_prediction_allows_numeric_values_outside_training_range_with_warning(self):
        payload = self.prediction_payload(avg_temperature_c=66)
        categories = {
            "region": ["Rangpur"],
            "year": [2020],
            "crop": ["Maize"],
            "season": ["Rabi"],
        }
        ranges = {column: (-100000, 100000) for column in main.NUMERIC}
        ranges["avg_temperature_c"] = (19.7, 34.0)

        with (
            patch.object(main, "metadata", {"dataset_loaded": True, "categories": categories}),
            patch.object(main, "imputer", SavedImputer()),
            patch.object(main, "feature_ranges", ranges),
            patch.object(main, "supported_crop_combinations", {("Rangpur", "Maize", "Rabi")}),
            patch.object(main, "_load_model", return_value=FittedEstimator()),
        ):
            result = main.predict(payload)

        self.assertEqual(set(result["results"]), {"TabPFN", "TabFM"})
        self.assertTrue(all(value["status"] == "ok" for value in result["results"].values()))
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("avg_temperature_c", result["warnings"][0])
        self.assertIn("19.7 to 34", result["warnings"][0])

    def test_prediction_runs_for_unseen_region_crop_season_with_warning(self):
        payload = self.prediction_payload(region="Chattogram", crop="Maize", season="Aus")
        categories = {
            "region": ["Rangpur", "Chattogram"],
            "year": [2020],
            "crop": ["Maize"],
            "season": ["Rabi", "Aus"],
        }
        ranges = {column: (-100000, 100000) for column in main.NUMERIC}
        estimators = {"TabPFN": FittedEstimator(), "TabFM": FittedEstimator()}

        with (
            patch.object(main, "metadata", {"dataset_loaded": True, "categories": categories}),
            patch.object(main, "imputer", SavedImputer()),
            patch.object(main, "feature_ranges", ranges),
            patch.object(main, "supported_crop_combinations", set()),
            patch.object(main, "_load_model", side_effect=estimators.__getitem__),
        ):
            result = main.predict(payload)

        self.assertEqual(set(result["results"]), {"TabPFN", "TabFM"})
        self.assertTrue(all(value["status"] == "ok" for value in result["results"].values()))
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("may be unreliable", result["warnings"][0])

    def test_negative_model_prediction_is_reported_as_invalid(self):
        class NegativeEstimator(FittedEstimator):
            def predict(self, _features):
                return np.array([-3.65])

        with patch.object(main, "_load_model", return_value=NegativeEstimator()):
            result = main._predict_one("TabPFN", None)

        self.assertEqual(result["status"], "error")
        self.assertIn("negative crop yield", result["message"])


if __name__ == "__main__":
    unittest.main()
