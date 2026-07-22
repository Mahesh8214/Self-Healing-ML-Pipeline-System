import os
import sys
import json
import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np


@dataclass
class DriftDetectorConfig:
    drift_report_dir = os.path.join('artifacts', 'reports', 'drift_report.json')

class DriftDetector:
    def __init__(self):
        self.drift_detector_config = DriftDetectorConfig()
    
    def calculate_psi(self, expected, actual, bins=10):
        # Remove NaN and infinite values
        expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
        actual = actual.replace([np.inf, -np.inf], np.nan).dropna()

        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        # Create bin edges from reference distribution
        bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)

        psi_values = []
        for e, a in zip(expected_percents, actual_percents):
            if e == 0:
                e = 0.0001
            if a == 0:
                a = 0.0001
            psi = (a - e) * np.log(a / e)
            psi_values.append(psi)

        return float(np.sum(psi_values))

    def calculate_categorical_psi(self, expected_series, actual_series):
        try:
            expected_counts = expected_series.value_counts(normalize=True)
            actual_counts = actual_series.value_counts(normalize=True)

            all_categories = set(expected_counts.index).union(set(actual_counts.index))
            psi = 0.0

            for cat in all_categories:
                e = expected_counts.get(cat, 0.0001)
                a = actual_counts.get(cat, 0.0001)
                if e == 0:
                    e = 0.0001
                if a == 0:
                    a = 0.0001
                psi += (a - e) * np.log(a / e)

            return float(psi)
        except Exception:
            return 0.0

    def detect_drift(self, reference_df, current_df):
        try:
            logging.info("Detecting data drift across numerical and categorical features")
            drift_results = {}

            numerical_columns = [col for col in ["carat", "depth", "table", "x", "y", "z"] if col in reference_df.columns and col in current_df.columns]
            categorical_columns = [col for col in ["cut", "color", "clarity"] if col in reference_df.columns and col in current_df.columns]

            overall_drift_detected = False

            # Process numerical columns
            for col in numerical_columns:
                ref_col = reference_df[col].dropna()
                cur_col = current_df[col].dropna()

                stat, p_value = ks_2samp(ref_col, cur_col)
                psi_score = self.calculate_psi(ref_col, cur_col)

                p_val_float = float(p_value)
                psi_float = float(psi_score)

                is_drifted = bool(p_val_float < 0.05 or psi_float > 0.20)

                if psi_float > 0.20 or p_val_float < 0.01:
                    status = "Drifted"
                elif psi_float > 0.10 or p_val_float < 0.05:
                    status = "Warning"
                else:
                    status = "Stable"

                drift_results[col] = {
                    "metric": "KS Test / PSI",
                    "ks_p_value": round(p_val_float, 4),
                    "psi_score": round(psi_float, 4),
                    "threshold": 0.20,
                    "status": status,
                    "drift_detected": is_drifted
                }

                if is_drifted:
                    overall_drift_detected = True

            # Process categorical columns
            for col in categorical_columns:
                psi_score = self.calculate_categorical_psi(reference_df[col], current_df[col])
                psi_float = float(psi_score)
                is_drifted = bool(psi_float > 0.20)

                if psi_float > 0.20:
                    status = "Drifted"
                elif psi_float > 0.10:
                    status = "Warning"
                else:
                    status = "Stable"

                drift_results[col] = {
                    "metric": "Categorical PSI",
                    "ks_p_value": None,
                    "psi_score": round(psi_float, 4),
                    "threshold": 0.20,
                    "status": status,
                    "drift_detected": is_drifted
                }

                if is_drifted:
                    overall_drift_detected = True

            return drift_results, overall_drift_detected

        except Exception as e:
            logging.error(f"Error in drift detection : {e}")
            raise e

    def initiate_drift_detection(self, reference_path, current_path):
        try:
            reference_df = pd.read_csv(reference_path)
            current_df = pd.read_csv(current_path)

            results, drift_flag = self.detect_drift(reference_df, current_df)

            drifted_features = [
                col for col, val in results.items()
                if val["drift_detected"]
            ]

            total_features = len(results)
            drifted_count = len(drifted_features)
            drift_percentage = round((drifted_count / total_features * 100), 1) if total_features > 0 else 0.0

            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "drift_detected": drift_flag,
                "drifted_features": drifted_features,
                "total_features": total_features,
                "drifted_count": drifted_count,
                "drift_percentage": drift_percentage,
                "retraining_required": drift_flag,
                "drift_threshold": 0.20,
                "feature_results": results
            }

            os.makedirs("artifacts/reports", exist_ok=True)

            with open(self.drift_detector_config.drift_report_dir, 'w') as f:
                json.dump(report, f, indent=4)

            logging.info(f"Drift report completed: {drifted_count}/{total_features} features drifted")

            return drift_flag
        except Exception as e:
            logging.error(f"Error in initiating drift detection : {e}")
            raise e


if __name__ == "__main__":
    detector = DriftDetector()
    drift = detector.initiate_drift_detection(
        "artifacts/data/reference_data.csv",
        "data/production_batches/batch_1.csv"
    )

    print("Drift detected:", drift)
