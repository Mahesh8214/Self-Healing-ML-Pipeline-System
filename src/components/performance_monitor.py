import sys
import pandas as pd
from sklearn.metrics import r2_score

from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from src.registry.model_registry import ModelRegistry


class PerformanceMonitor:

    def evaluate_model(self, batch_path):
        try:
            df = pd.read_csv(batch_path)
            target = "price"

            X = df.drop(columns=[target])
            y = df[target]

            registry = ModelRegistry()
            model_path = registry.get_latest_model()
            model = load_object(model_path)
            preprocessor = load_object("artifacts/preprocessor.pkl")

            X_transformed = preprocessor.transform(X)
            predictions = model.predict(X_transformed)

            score = r2_score(y, predictions)
            logging.info(f"Production batch R2 score: {score}")

            return score

        except Exception as e:
            raise CustomException(e, sys)