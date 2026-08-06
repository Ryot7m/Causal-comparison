import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.ateplot import plot_if_guard_seg
from workspace.drcdfplot import drcdf_plot


def test_drcdf_plot_saves_images(tmp_path):
    data = pd.DataFrame({
        "seg": [0, 0, 1, 1],
        "c": [0, 1, 0, 1],
        "threshold": [1.0, 2.0, 1.0, 2.0],
        "F1_dr": [0.4, 1.0, 0.3, 1.0],
        "F0_dr": [0.5, 1.0, 0.4, 1.0],
        "tau_c": [-0.1, 0.0, -0.1, 0.0],
        "ci_low": [-0.2, 0.0, -0.2, 0.0],
        "ci_high": [0.0, 0.0, 0.0, 0.0],
    })

    output_dir = tmp_path / "drcdf"

    saved_paths = drcdf_plot(
        cdf_seg=data,
        output_dir=output_dir,
        show=False,
    )

    assert len(saved_paths) == 4
    assert all(path.exists() for path in saved_paths)
    assert all(path.stat().st_size > 0 for path in saved_paths)
    assert plt.get_fignums() == []


def test_ate_plot_saves_each_segment_image(tmp_path):
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    outcome = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    segment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score_values = np.array([0.0, 1.0])

    nuisance = {
        "e_hat": np.full(8, 0.5),
        "mu1_hat": np.full(8, 0.7),
        "mu0_hat": np.full(8, 0.3),
    }

    output_dir = tmp_path / "ate"

    result = plot_if_guard_seg(
        A=treatment,
        Y=outcome,
        s_values=score_values,
        nuis=nuisance,
        seg=segment,
        cap=None,
        B=50,
        seed=123,
        bins=10,
        save_dir=output_dir,
    )

    expected_paths = [
        output_dir / "if_guard_seg0.png",
        output_dir / "if_guard_seg1.png",
    ]

    assert len(result) == 2
    assert set(result["seg"]) == {0, 1}
    assert all(path.exists() for path in expected_paths)
    assert all(path.stat().st_size > 0 for path in expected_paths)
    assert plt.get_fignums() == []