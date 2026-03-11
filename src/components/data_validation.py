import os
import sys
import pandas as pd
import json

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataValidationConfig:
    validation_report_path = os.path.join(
        "artifacts",
        "reports",
        "validation_report.json"
    )


class DataValidation:

    def __init__(self):
        self.validation_config = DataValidationConfig()


    def validate_schema(self, df):
        expected_columns = [
            "carat",
            "cut",
            "color",
            "clarity",
            "depth",
            "table",
            "x",
            "y",
            "z",
            "price"
        ]

        for col in expected_columns:
            if col not in df.columns:
                return False

        return True


    def check_missing_values(self, df):
        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            return False

        return True


    def initiate_data_validation(self, file_path):
        try:
            logging.info("Starting data validation")
            df = pd.read_csv(file_path)
            schema_valid = self.validate_schema(df)
            missing_valid = self.check_missing_values(df)
            validation_status = schema_valid and missing_valid

            validation_report = {
                "schema_valid": schema_valid,
                "missing_values_valid": missing_valid,
                "validation_status": validation_status
            }

            os.makedirs(
                os.path.dirname(self.validation_config.validation_report_path),
                exist_ok=True
            )

            with open(self.validation_config.validation_report_path,"w") as f:
                json.dump(validation_report,f,indent=4)

            logging.info("Validation report generated")

            return validation_status

        except Exception as e:
            raise CustomException(e,sys)