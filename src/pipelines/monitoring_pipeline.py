import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.data_validation import DataValidation
from src.components.drift_detector import DriftDetector
from src.components.performance_monitor import PerformanceMonitor

# utility functions added in utils.py
from src.utils import log_monitoring
from src.logger import (
    is_batch_processed,
    mark_batch_processed
)


class MonitoringPipeline:

    def __init__(self):
        # Reference dataset used for drift comparison
        self.reference_path = "artifacts/data/reference_data.csv"

        # Folder containing simulated production batches
        self.production_folder = "data/production_batches"


    def run_monitoring(self):
        try:
            logging.info("Monitoring pipeline started")
            batches = sorted(os.listdir(self.production_folder))
            for batch_file in batches:
                # -----------------------------------
                # Step 0 — Skip already processed batches
                # -----------------------------------
                if is_batch_processed(batch_file):
                    logging.info(f"Skipping already processed batch: {batch_file}")
                    continue

                batch_path = os.path.join(self.production_folder, batch_file)
                logging.info(f"Processing batch: {batch_file}")
                # -----------------------------------
                # Step 1 — Data Validation
                # -----------------------------------
                validation = DataValidation()
                status = validation.initiate_data_validation(batch_path)
                if not status:
                    logging.warning(f"Data validation failed for batch: {batch_file}")
                    continue
                # -----------------------------------
                # Step 2 — Drift Detection
                # ----------------------------------
                detector = DriftDetector()
                drift = detector.initiate_drift_detection(
                    self.reference_path,
                    batch_path
                )
                logging.info(f"Drift result for {batch_file}: {drift}")
                # -----------------------------------
                # Step 3 — Model Performance Check
                # -----------------------------------
                monitor = PerformanceMonitor()
                score = monitor.evaluate_model(batch_path)
                logging.info(f"Batch R2 Score: {score}")
                # -----------------------------------
                # Step 4 — Retraining Decision
                # -----------------------------------
                retraining_triggered = False

                if drift and score < 0.8:
                    logging.warning("Drift + Performance drop → Retraining model")
                    # Import here to avoid circular import
                    from src.pipelines.training_pipeline import run_training_pipeline
                    run_training_pipeline()
                    retraining_triggered = True
                else:
                    logging.info("No retraining required")
                # -----------------------------------
                # Step 5 — Save Monitoring Log
                # -----------------------------------
                log_monitoring(
                    batch=batch_file,
                    drift=drift,
                    score=score,
                    retrained=retraining_triggered
                )
                # -----------------------------------
                # Step 6 — Mark batch as processed
                # -----------------------------------
                mark_batch_processed(batch_file)
            logging.info("Monitoring pipeline completed")
        except Exception as e:
            logging.error(f"Error in monitoring pipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    monitor = MonitoringPipeline()
    monitor.run_monitoring()