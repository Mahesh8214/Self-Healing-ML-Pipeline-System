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
from sklearn.metrics import r2_score
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

            # Evaluate new model
            y_pred_new = best_model.predict(X_test)
            new_score = r2_score(y_test, y_pred_new)

            logging.info(f"New model R2 score: {new_score}")

            registry = ModelRegistry()

            old_model_path = registry.get_latest_model()

            deploy_new_model = True

            if old_model_path is not None:

                try:
                    old_model = load_object(old_model_path)

                    y_pred_old = old_model.predict(X_test)

                    old_score = r2_score(y_test, y_pred_old)

                    logging.info(f"Previous model R2 score: {old_score}")

                    if new_score <= old_score:
                        logging.info("New model worse → rollback (keeping old model)")
                        deploy_new_model = False
                except Exception as e:
                    logging.warning("Could not evaluate old model, deploying new model")

            if deploy_new_model:
                version = registry.get_next_version()

                model_path = os.path.join(
                    self.model_trainer_config.trained_model_dir,
                    f"model_{version}.pkl"
                )

                os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)

                save_object(
                    file_path=model_path,
                    obj=best_model
                )

                registry.register_model(model_path, reason="performance_degradation_after_drift")

                logging.info(f"New model deployed at {model_path}")

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)