import json

import numpy as np

from app.renew import to_json_compatible

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