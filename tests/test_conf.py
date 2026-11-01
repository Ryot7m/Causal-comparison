import pytest

from app.database import Base, engine
from app.models import AnalysisJob  # モデルをBaseに登録する


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    Base.metadata.create_all(bind=engine)
    yield