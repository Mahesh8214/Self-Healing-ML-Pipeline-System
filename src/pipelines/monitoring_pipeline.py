import os
import sys
import json
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.data_validation import DataValidation
from src.components.drift_detector import DriftDetector
from src.components.performance_monitor import PerformanceMonitor

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

                from src.job_manager import JobManager
                JobManager.log_event(
                    event_type="MONITORING_STARTED",
                    description=f"Monitoring pipeline started for batch {batch_file}",
                    details={"batch": batch_file}
                )

                # -----------------------------------
                # Step 1 — Data Validation
                # -----------------------------------
                validation = DataValidation()
                status = validation.initiate_data_validation(batch_path)
                if not status:
                    logging.warning(f"Data validation failed for batch: {batch_file}")
                    JobManager.log_event(
                        event_type="VALIDATION_FAILED",
                        description=f"Data validation failed for batch {batch_file}",
                        details={"batch": batch_file}
                    )
                    continue

                JobManager.log_event(
                    event_type="VALIDATION_PASSED",
                    description=f"Data validation passed for batch {batch_file}",
                    details={"batch": batch_file}
                )

                # -----------------------------------
                # Step 2 — Drift Detection
                # ----------------------------------
                detector = DriftDetector()
                drift = detector.initiate_drift_detection(
                    self.reference_path,
                    batch_path
                )
                logging.info(f"Drift result for {batch_file}: {drift}")

                # Load detailed drift report for log
                try:
                    with open("artifacts/reports/drift_report.json", "r") as rf:
                        rep = json.load(rf)
                    drifted_cols = rep.get("drifted_features", [])
                except Exception:
                    drifted_cols = []

                if drift:
                    JobManager.log_event(
                        event_type="DRIFT_DETECTED",
                        description=f"Data drift detected in batch {batch_file}. Drifted features: {', '.join(drifted_cols)}",
                        details={"batch": batch_file, "drifted_features": drifted_cols}
                    )
                else:
                    JobManager.log_event(
                        event_type="NO_DRIFT",
                        description=f"No data drift detected in batch {batch_file}",
                        details={"batch": batch_file}
                    )

                # -----------------------------------
                # Step 3 — Model Performance Check
                # -----------------------------------
                monitor = PerformanceMonitor()
                score = monitor.evaluate_model(batch_path)
                logging.info(f"Batch R2 Score: {score}")

                JobManager.log_event(
                    event_type="PERFORMANCE_CHECKED",
                    description=f"Production batch R2 score evaluated: {score:.4f}",
                    details={"batch": batch_file, "r2_score": score}
                )

                # -----------------------------------
                # Step 4 — Retraining Decision
                # -----------------------------------
                retraining_triggered = False
                settings = JobManager.get_settings()
                auto_healing = settings.get("auto_healing", True)
                demo_mode = settings.get("demo_mode", True)

                # Drift condition: drift is detected or score drops below threshold (0.85)
                if drift or score < 0.85:
                    logging.warning(f"Drift or Performance Drop detected in {batch_file}")

                    if auto_healing:
                        if not JobManager.has_active_job():
                            job = JobManager.create_job(trigger_reason=f"auto_healing_drift_{batch_file}")
                            if job:
                                logging.info(f"Triggering background retraining job #{job['job_id']}")
                                from src.pipelines.training_pipeline import run_training_pipeline
                                import threading

                                thread = threading.Thread(
                                    target=run_training_pipeline,
                                    kwargs={"job_id": job["job_id"], "demo_mode": demo_mode},
                                    daemon=True
                                )
                                thread.start()
                                retraining_triggered = True
                        else:
                            logging.info("Active training job already in progress. Skipping duplicate trigger.")
                    else:
                        logging.info("Auto-Healing is OFF. Retraining recommended.")
                        JobManager.log_event(
                            event_type="RETRAINING_RECOMMENDED",
                            description=f"Retraining recommended for batch {batch_file} (Auto-Healing OFF)",
                            details={"batch": batch_file}
                        )

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
