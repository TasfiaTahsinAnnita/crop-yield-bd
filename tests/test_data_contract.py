"""Small deterministic checks for the supplied dataset contract."""
import json
import unittest
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "bangladesh_crop_yield_synthetic_1000.csv"
TARGET = "crop_yield_tonnes_ha"
FEATURES = [
    "region", "year", "crop", "season", "rainfall_mm_season",
    "avg_temperature_c", "relative_humidity_pct", "soil_ph",
    "soil_moisture_m3_m3", "solar_radiation_mj_m2_day", "ndvi",
    "fertilizer_kg_ha", "irrigation_used_0_no_1_yes",
    "previous_season_yield_tonnes_ha",
]


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV)

    def test_target_features_and_metadata(self):
        self.assertEqual(len(self.df), 1000)
        self.assertTrue(set(FEATURES + [TARGET]).issubset(self.df.columns))
        self.assertTrue({"record_id", "country", "data_status"}.issubset(self.df.columns))
        self.assertNotIn("data_status", FEATURES)
        self.assertEqual(self.df["data_status"].unique().tolist(), ["SYNTHETIC_FOR_PROTOTYPE"])

    def test_categories_and_irrigation_encoding(self):
        self.assertEqual(set(self.df["irrigation_used_0_no_1_yes"].unique()), {0, 1})
        self.assertEqual(len(self.df["region"].unique()), 9)
        self.assertEqual(set(self.df["crop"].unique()), {"Rice", "Wheat", "Maize", "Jute", "Potato"})
        self.assertEqual(set(self.df["season"].unique()), {"Kharif", "Rabi", "Boro", "Aus", "Aman"})

    def test_seed_42_split_is_reproducible_and_80_20(self):
        indices = list(range(len(self.df)))
        train_a, test_a = train_test_split(indices, test_size=0.2, random_state=42)
        train_b, test_b = train_test_split(indices, test_size=0.2, random_state=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual((len(train_a), len(test_a)), (800, 200))


if __name__ == "__main__":
    unittest.main()
