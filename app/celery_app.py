import os

from celery import Celery

celery_app = Celery(
    "causal_comparison",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json"
)

celery_app.conf.beat_schedule = {
    "recover-stale-analysis-jobs-every-5-minutes": {
        "task": "app.tasks.recover_stale_analysis_jobs",
        "schedule": 300.0,
        "kwargs": {
            "stale_after_minutes": 60,
        },
    },
}