import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation


def run_training_pipeline(job_id=None, demo_mode=False):
    from src.job_manager import JobManager
    try:
        logging.info(f"Training pipeline started. Job ID: {job_id}, Demo Mode: {demo_mode}")

        if job_id:
            JobManager.update_job_stage(job_id, "Data Validation", "RUNNING")

        reference_path = "artifacts/data/reference_data.csv"
        validation = DataValidation()
        status = validation.initiate_data_validation(reference_path)
        if not status:
            raise Exception("Data validation failed for reference dataset")

        if job_id:
            JobManager.update_job_stage(job_id, "Data Validation", "COMPLETED")
            JobManager.update_job_stage(job_id, "Drift Detection", "COMPLETED")
            JobManager.update_job_stage(job_id, "Retraining Trigger", "COMPLETED")
            JobManager.update_job_stage(job_id, "Model Training", "RUNNING")

        df = pd.read_csv(reference_path)
        logging.info("Reference dataset loaded")

        if demo_mode and len(df) > 1000:
            df = df.sample(n=1000, random_state=42).reset_index(drop=True)
            logging.info("Demo mode active: Dataset sampled to 1000 rows for fast training")

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        os.makedirs("artifacts/tmp", exist_ok=True)
        train_path = "artifacts/tmp/train.csv"
        test_path = "artifacts/tmp/test.csv"

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        logging.info("Train test split completed")

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path,
            test_path
        )
        logging.info("Data transformation completed")

        trainer = ModelTrainer()
        res = trainer.initiate_model_training(
            train_arr,
            test_arr
        )

        if job_id:
            JobManager.update_job_stage(job_id, "Model Evaluation", "COMPLETED")
            JobManager.update_job_stage(job_id, "Champion vs Challenger", "COMPLETED")
            JobManager.update_job_stage(job_id, "Quality Gate", "COMPLETED")
            JobManager.update_job_stage(job_id, "Model Promotion", "COMPLETED")

            JobManager.complete_job(
                job_id=job_id,
                model_version=res.get("version"),
                champion_metrics=res.get("champion_metrics"),
                challenger_metrics=res.get("challenger_metrics"),
                promotion_decision=res.get("promotion_decision"),
                promotion_reason=res.get("promotion_reason")
            )

        logging.info(f"Training pipeline completed successfully. Decision: {res.get('promotion_decision')}")
        return res

    except Exception as e:
        logging.error(f"Training pipeline error: {e}")
        if job_id:
            JobManager.fail_job(job_id, str(e))
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
 