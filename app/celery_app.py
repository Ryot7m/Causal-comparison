import os

from app.celery_app import Celery

celery_app = Celery(
    "causal_comparison",
    broker=os.getenv(
        "CELERY_BROKER_URL",
        "redis://redis:6379/0",
    ),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)