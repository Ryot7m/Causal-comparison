import json

import numpy as np

from datetime import datetime, timedelta, timezone

from app.database import SessionLocal
from app.models import AnalysisJob
from app.renew import create_analysis_job, recover_stale_analysis_jobs, to_json_compatible
import app.storage as astg

def test_to_json_compatible_converts_numpy_values():
    value = {
        "cut": np.int64(1),
        "values": np.array([2, 3]),
        "nested": {"count": np.int64(4)}
    }

    converted = to_json_compatible(value)
    assert converted == {
        "cut": 1,
        "values": [2, 3],
        "nested": {"count": 4}
    }
    # DBのJSONB保存時に使われるJSON変換でエラーにならないことも確認
    json.dumps(converted)
    
def test_recover_stale_analysis_jobs_only_fails_old_running(
    tmp_path,
    monkeypatch,
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(astg, "UPLOAD_DIR", upload_dir)

    old_csv = upload_dir / "old.csv"
    old_csv.write_text("x\n1\n", encoding="utf-8")

    now = datetime.now(timezone.utc)
    stale_before = now - timedelta(minutes=60)

    with SessionLocal() as db:
        old_job = create_analysis_job(
            db=db,
            config={},
            filename="old.csv",
            input_path=str(old_csv),
        )
        old_job.status = "running"
        old_job.started_at = stale_before - timedelta(minutes=1)

        recent_job = create_analysis_job(
            db=db,
            config={},
            filename="recent.csv",
        )
        recent_job.status = "running"
        recent_job.started_at = now - timedelta(minutes=5)

        queued_job = create_analysis_job(
            db=db,
            config={},
            filename="queued.csv",
        )

        old_id = old_job.id
        recent_id = recent_job.id
        queued_id = queued_job.id
        db.commit()

        recovered_count = recover_stale_analysis_jobs(
            db,
            stale_before=stale_before,
        )

    assert recovered_count == 1
    assert not old_csv.exists()

    with SessionLocal() as db:
        old_job = db.get(AnalysisJob, old_id)
        recent_job = db.get(AnalysisJob, recent_id)
        queued_job = db.get(AnalysisJob, queued_id)

        assert old_job.status == "failed"
        assert "規定時間内" in old_job.error_message
        assert old_job.finished_at is not None
        assert old_job.input_path is None

        assert recent_job.status == "running"
        assert queued_job.status == "queued"