import os
import json
import uuid
import threading
from datetime import datetime
from src.logger import logging

JOBS_PATH = "artifacts/metadata/training_jobs.json"
TIMELINE_PATH = "artifacts/monitoring/timeline_events.json"
SETTINGS_PATH = "artifacts/metadata/settings.json"

PIPELINE_STAGES = [
    "Data Validation",
    "Drift Detection",
    "Retraining Trigger",
    "Model Training",
    "Model Evaluation",
    "Champion vs Challenger",
    "Quality Gate",
    "Model Promotion"
]

class JobManager:
    _lock = threading.Lock()

    @staticmethod
    def _ensure_files():
        os.makedirs("artifacts/metadata", exist_ok=True)
        os.makedirs("artifacts/monitoring", exist_ok=True)

        if not os.path.exists(JOBS_PATH):
            with open(JOBS_PATH, "w") as f:
                json.dump({"jobs": []}, f, indent=4)

        if not os.path.exists(TIMELINE_PATH):
            with open(TIMELINE_PATH, "w") as f:
                json.dump({"events": []}, f, indent=4)

        if not os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "w") as f:
                json.dump({
                    "auto_healing": True,
                    "demo_mode": True,
                    "min_improvement_threshold": 0.001
                }, f, indent=4)

    @classmethod
    def get_settings(cls):
        cls._ensure_files()
        with cls._lock:
            try:
                with open(SETTINGS_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading settings: {e}")
                return {"auto_healing": True, "demo_mode": True, "min_improvement_threshold": 0.001}

    @classmethod
    def update_settings(cls, auto_healing=None, demo_mode=None, min_improvement_threshold=None):
        cls._ensure_files()
        with cls._lock:
            settings = cls.get_settings()
            if auto_healing is not None:
                settings["auto_healing"] = bool(auto_healing)
            if demo_mode is not None:
                settings["demo_mode"] = bool(demo_mode)
            if min_improvement_threshold is not None:
                settings["min_improvement_threshold"] = float(min_improvement_threshold)

            with open(SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=4)
            return settings

    @classmethod
    def load_jobs(cls):
        cls._ensure_files()
        with cls._lock:
            try:
                with open(JOBS_PATH, "r") as f:
                    return json.load(f).get("jobs", [])
            except Exception as e:
                logging.error(f"Error loading jobs: {e}")
                return []

    @classmethod
    def save_jobs(cls, jobs):
        cls._ensure_files()
        with cls._lock:
            with open(JOBS_PATH, "w") as f:
                json.dump({"jobs": jobs}, f, indent=4)

    @classmethod
    def has_active_job(cls):
        jobs = cls.load_jobs()
        for job in jobs:
            if job.get("status") in ["QUEUED", "RUNNING"]:
                return True
        return False

    @classmethod
    def get_latest_job(cls):
        jobs = cls.load_jobs()
        if jobs:
            return jobs[-1]
        return None

    @classmethod
    def get_job(cls, job_id):
        jobs = cls.load_jobs()
        for job in jobs:
            if job.get("job_id") == job_id:
                return job
        return None

    @classmethod
    def create_job(cls, trigger_reason="drift_detected"):
        if cls.has_active_job():
            logging.warning("Job creation rejected: active job already in progress")
            return None

        cls._ensure_files()
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        initial_stages = {stage: "WAITING" for stage in PIPELINE_STAGES}

        job = {
            "job_id": job_id,
            "status": "QUEUED",
            "current_stage": "Data Validation",
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "model_version": None,
            "trigger_reason": trigger_reason,
            "stages": initial_stages,
            "champion_metrics": None,
            "challenger_metrics": None,
            "promotion_decision": None,
            "promotion_reason": None
        }

        jobs = cls.load_jobs()
        jobs.append(job)
        cls.save_jobs(jobs)

        cls.log_event(
            event_type="JOB_CREATED",
            description=f"Retraining job #{job_id} queued. Trigger: {trigger_reason}",
            details={"job_id": job_id}
        )

        return job

    @classmethod
    def update_job_stage(cls, job_id, stage_name, stage_status="RUNNING", details=None):
        jobs = cls.load_jobs()
        for job in jobs:
            if job["job_id"] == job_id:
                job["current_stage"] = stage_name
                if stage_status == "RUNNING" and job["status"] == "QUEUED":
                    job["status"] = "RUNNING"
                    job["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if "stages" not in job:
                    job["stages"] = {stage: "WAITING" for stage in PIPELINE_STAGES}

                # Mark all previous stages as COMPLETED if current stage is moving past them
                stage_idx = PIPELINE_STAGES.index(stage_name) if stage_name in PIPELINE_STAGES else -1
                for idx, s_name in enumerate(PIPELINE_STAGES):
                    if idx < stage_idx and job["stages"].get(s_name) != "COMPLETED":
                        job["stages"][s_name] = "COMPLETED"
                    elif idx == stage_idx:
                        job["stages"][s_name] = stage_status

                if details:
                    job.update(details)

                cls.save_jobs(jobs)

                cls.log_event(
                    event_type="STAGE_UPDATE",
                    description=f"Job #{job_id} stage '{stage_name}' set to {stage_status}",
                    details={"job_id": job_id, "stage": stage_name, "status": stage_status}
                )
                break

    @classmethod
    def complete_job(cls, job_id, model_version=None, champion_metrics=None, challenger_metrics=None, promotion_decision=None, promotion_reason=None):
        jobs = cls.load_jobs()
        for job in jobs:
            if job["job_id"] == job_id:
                job["status"] = "COMPLETED"
                job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job["model_version"] = model_version
                job["champion_metrics"] = champion_metrics
                job["challenger_metrics"] = challenger_metrics
                job["promotion_decision"] = promotion_decision
                job["promotion_reason"] = promotion_reason

                for s_name in PIPELINE_STAGES:
                    job["stages"][s_name] = "COMPLETED"

                cls.save_jobs(jobs)

                cls.log_event(
                    event_type="JOB_COMPLETED",
                    description=f"Job #{job_id} completed successfully. Decision: {promotion_decision}",
                    details={
                        "job_id": job_id,
                        "model_version": model_version,
                        "promotion_decision": promotion_decision,
                        "promotion_reason": promotion_reason
                    }
                )
                break

    @classmethod
    def fail_job(cls, job_id, error_message):
        jobs = cls.load_jobs()
        for job in jobs:
            if job["job_id"] == job_id:
                job["status"] = "FAILED"
                job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job["error_message"] = str(error_message)

                current = job.get("current_stage")
                if current and "stages" in job:
                    job["stages"][current] = "FAILED"

                cls.save_jobs(jobs)

                cls.log_event(
                    event_type="JOB_FAILED",
                    description=f"Job #{job_id} failed: {error_message}",
                    details={"job_id": job_id, "error": str(error_message)}
                )
                break

    @classmethod
    def log_event(cls, event_type, description, details=None):
        cls._ensure_files()
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "description": description,
            "details": details or {}
        }
        with cls._lock:
            try:
                with open(TIMELINE_PATH, "r") as f:
                    data = json.load(f)
                data.setdefault("events", []).append(event)
                with open(TIMELINE_PATH, "w") as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                logging.error(f"Error saving timeline event: {e}")

    @classmethod
    def get_timeline_events(cls):
        cls._ensure_files()
        with cls._lock:
            try:
                with open(TIMELINE_PATH, "r") as f:
                    return json.load(f).get("events", [])
            except Exception as e:
                logging.error(f"Error loading timeline events: {e}")
                return []
