import logging 
import os
import json
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs",  LOG_FILE)
os.makedirs(logs_path, exist_ok = True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)


BATCH_LOG_PATH = "artifacts/monitoring/batch_log.json"
def load_batch_log():
    os.makedirs("artifacts/monitoring", exist_ok=True)
    if not os.path.exists(BATCH_LOG_PATH):
        with open(BATCH_LOG_PATH, "w") as f:
            json.dump({"processed_batches": []}, f)
    with open(BATCH_LOG_PATH, "r") as f:
        return json.load(f)


def save_batch_log(data):
    with open(BATCH_LOG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def is_batch_processed(batch_name):
    data = load_batch_log()
    return batch_name in data["processed_batches"]


def mark_batch_processed(batch_name):
    data = load_batch_log()
    data["processed_batches"].append(batch_name)
    save_batch_log(data)