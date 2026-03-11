import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation


def run_training_pipeline():

    try:
        logging.info("Training pipeline started")

        # Load reference dataset
        reference_path = "artifacts/data/reference_data.csv"
        validation = DataValidation()
        status = validation.initiate_data_validation(reference_path)
        if not status:
            raise Exception("Data validation failed")

        df = pd.read_csv(reference_path)

        logging.info("Reference dataset loaded")

        # Train/Test split
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        # Save temporary splits
        os.makedirs("artifacts/tmp", exist_ok=True)

        train_path = "artifacts/tmp/train.csv"
        test_path = "artifacts/tmp/test.csv"

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        logging.info("Train test split completed")

        # Data Transformation
        transformation = DataTransformation()

        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        logging.info("Data transformation completed")

        # Model Training
        trainer = ModelTrainer()

        trainer.initiate_model_training(
            train_arr,
            test_arr
        )

        logging.info("Training pipeline completed")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline() 