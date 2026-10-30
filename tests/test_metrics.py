"""Checks for held-out regression metric calculations."""
import unittest

import numpy as np

from python_backend.main import _calculate_metrics


class RegressionMetricTests(unittest.TestCase):
    def test_metrics_include_mean_squared_error(self):
        metrics = _calculate_metrics(
            np.array([1.0, 3.0]),
            np.array([2.0, 2.0]),
        )

        self.assertEqual(metrics["MSE"], 1.0)
        self.assertEqual(metrics["RMSE"], 1.0)
        self.assertEqual(metrics["MAE"], 1.0)
        self.assertEqual(metrics["R2"], 0.0)


if __name__ == "__main__":
    unittest.main()
