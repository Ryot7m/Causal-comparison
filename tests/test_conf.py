import pytest

from app.database import Base, engine
import app.models  # モデルをBaseに登録する


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    Base.metadata.create_all(bind=engine)
    yield