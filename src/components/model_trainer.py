# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.registry.model_registry import ModelRegistry

from src.utils import save_object
from src.utils import evaluate_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.utils import load_object


from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_dir = os.path.join('artifacts','models')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            # Train the selected best model
            best_model.fit(X_train, y_train)

            # Evaluate candidate Challenger model
            y_pred_challenger = best_model.predict(X_test)
            challenger_r2 = float(r2_score(y_test, y_pred_challenger))
            challenger_mae = float(mean_absolute_error(y_test, y_pred_challenger))
            challenger_mse = float(mean_squared_error(y_test, y_pred_challenger))
            challenger_rmse = float(np.sqrt(challenger_mse))

            challenger_metrics = {
                "r2": round(challenger_r2, 4),
                "mae": round(challenger_mae, 2),
                "mse": round(challenger_mse, 2),
                "rmse": round(challenger_rmse, 2),
                "model_name": best_model_name
            }

            logging.info(f"Challenger Model ({best_model_name}) Metrics: {challenger_metrics}")

            registry = ModelRegistry()
            old_model_path = registry.get_latest_model()

            deploy_new_model = True
            champion_metrics = None
            promotion_reason = ""
            champion_version = "v0"

            from src.job_manager import JobManager
            settings = JobManager.get_settings()
            min_improvement = settings.get("min_improvement_threshold", 0.001)

            if old_model_path is not None and os.path.exists(old_model_path):
                try:
                    old_model = load_object(old_model_path)
                    y_pred_champion = old_model.predict(X_test)

                    champion_r2 = float(r2_score(y_test, y_pred_champion))
                    champion_mae = float(mean_absolute_error(y_test, y_pred_champion))
                    champion_mse = float(mean_squared_error(y_test, y_pred_champion))
                    champion_rmse = float(np.sqrt(champion_mse))

                    reg_data = registry.load_registry()
                    champion_version = f"v{len(reg_data.get('versions', []))}" if reg_data.get("versions") else "v1"

                    champion_metrics = {
                        "r2": round(champion_r2, 4),
                        "mae": round(champion_mae, 2),
                        "mse": round(champion_mse, 2),
                        "rmse": round(champion_rmse, 2),
                        "model_version": champion_version
                    }

                    logging.info(f"Champion Model Metrics: {champion_metrics}")

                    # Quality Gate evaluation
                    r2_diff = challenger_r2 - champion_r2
                    if challenger_r2 >= (champion_r2 + min_improvement):
                        deploy_new_model = True
                        promotion_reason = (
                            f"Challenger promoted because R2 improved from {champion_r2:.4f} to {challenger_r2:.4f} "
                            f"(+{r2_diff:.4f}, exceeding required min delta {min_improvement:.4f})."
                        )
                    else:
                        deploy_new_model = False
                        promotion_reason = (
                            f"Challenger rejected because R2 score ({challenger_r2:.4f}) did not exceed "
                            f"Champion R2 ({champion_r2:.4f}) + min threshold ({min_improvement:.4f})."
                        )
                except Exception as e:
                    logging.warning(f"Could not evaluate existing model: {e}. Defaulting to deploy new model.")
                    deploy_new_model = True
                    promotion_reason = "No valid prior Champion model available for comparison; deploying candidate model."
            else:
                promotion_reason = "Initial production model deployment."

            assigned_version = "v1"
            if deploy_new_model:
                assigned_version = registry.get_next_version()

                model_path = os.path.join(
                    self.model_trainer_config.trained_model_dir,
                    f"model_{assigned_version}.pkl"
                )

                os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)

                save_object(
                    file_path=model_path,
                    obj=best_model
                )

                registry.register_model(model_path, reason="performance_degradation_after_drift")

                logging.info(f"New model version {assigned_version} deployed at {model_path}")
            else:
                assigned_version = champion_version
                logging.info(f"Model deployment skipped. Active model remains {champion_version}.")

            return {
                "deploy_new_model": deploy_new_model,
                "version": assigned_version,
                "best_model_name": best_model_name,
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
                "promotion_decision": "PROMOTED" if deploy_new_model else "REJECTED",
                "promotion_reason": promotion_reason
            }

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)