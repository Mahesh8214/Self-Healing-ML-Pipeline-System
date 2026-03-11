import os
import sys
import json
import pandas as pd
from scipy.stats import ks_2samp

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np


@dataclass
class DriftDetectorConfig:
    drift_report_dir = os.path.join('artifacts','reports','drift_report.json')

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

    def detect_drift(self, reference_df,current_df):
        try:
            logging.info("Detecting data drift ")
            drift_results ={}

            numerical_columns = [
                "carat",
                "depth",
                "table",
                "x",
                "y",
                "z",
            ]
            drift_detected = False

            for col in numerical_columns:
                stat, p_value = ks_2samp(reference_df[col], current_df[col])
                psi_score = self.calculate_psi(reference_df[col], current_df[col])

                drift = bool(p_value < 0.05 or psi_score > 0.2)
                drift_results[col] = {
                    "ks_p_value": float(p_value),
                    "psi_score": float(psi_score),
                    "drift_detected": drift
                }
                if drift:
                    drift_detected = True

            return drift_results, drift_detected

        except Exception as e:
            logging.error(f"Error in drift detection : {e}")
            raise e

    def initiate_drift_detection(self, reference_path, current_path):
        try:
            reference_df = pd.read_csv(reference_path)
            current_df = pd.read_csv(current_path)

            results, drift_flag = self.detect_drift(reference_df,current_df)

            drifted_features = [
                col for col, val in results.items()
                if val["drift_detected"]
            ]

            report = {
                "drift_detected": drift_flag,
                "drifted_features": drifted_features,
                "total_features": len(results),
                "feature_results": results
            }

            os.makedirs("artifacts/reports", exist_ok=True)

            with open(self.drift_detector_config.drift_report_dir,'w') as f:
                json.dump(report,f,indent=4)
            
            logging.info("Drift report completed")

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