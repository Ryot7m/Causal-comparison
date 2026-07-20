from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///analysis.db"

engine = create_engine(DATABASE_URL)