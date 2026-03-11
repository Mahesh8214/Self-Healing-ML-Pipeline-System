import os
import json
from datetime import datetime

REGISTRY_PATH = "artifacts/metadata/model_registry.json"
MODEL_DIR = "artifacts/models"

class ModelRegistry:
    def __init__(self) :
        if not os.path.exists(REGISTRY_PATH):
            os.makedirs(os.path.dirname(REGISTRY_PATH),exist_ok=True) 
            with open(REGISTRY_PATH, 'w') as f:
                json.dump({}, f)
    
    def load_registry(self):
        with open(REGISTRY_PATH,'r') as f:
            registry = json.load(f)
            return registry
    
    def save_registry(self,registry):
        with open(REGISTRY_PATH,'w') as f:
            json.dump(registry,f,indent=4)
    
    def get_next_version(self):
        registry = self.load_registry()
        version = len(registry['versions']) + 1
        return f'v{version}'
    
    def register_model(self, model_path, reason="manual_training"):

        registry = self.load_registry()
        version = self.get_next_version()

        entry = {
            "version": version,
            "model_path": model_path,
            "timestamp": str(datetime.now()),
            "reason": reason
        }

        registry["versions"].append(entry)
        registry["latest_model"] = model_path
        self.save_registry(registry)    

    def get_latest_model(self):
        registry = self.load_registry()

        return registry['latest_model']

