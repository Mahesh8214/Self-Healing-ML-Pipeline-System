import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
import json

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    


MONITOR_LOG_PATH = "artifacts/monitoring/monitoring_log.json"

def log_monitoring(batch, drift, score, retrained):

    import os
    import json
    from datetime import datetime

    log_path = "artifacts/monitoring/monitoring_log.json"

    os.makedirs("artifacts/monitoring", exist_ok=True)

    entry = {
        "timestamp": str(datetime.now()),
        "batch": batch,
        "drift_detected": drift,
        "r2_score": score,
        "retraining_triggered": retrained
    }

    if os.path.exists(log_path):

        with open(log_path) as f:
            data = json.load(f)

    else:
        data = []

    data.append(entry)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)