import os

from celery import Celery

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