"""Generate reproducible CSV fixtures for the API and research pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).parent
ROWS_PER_SEGMENT = 120
SEED = 42


def _base_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    within_segment = np.tile(np.arange(ROWS_PER_SEGMENT), 3)
    segment = np.repeat([1, 2, 3], ROWS_PER_SEGMENT)
    return within_segment, segment


def generate_api_sample() -> pd.DataFrame:
    within_segment, segment = _base_data()
    rng = np.random.default_rng(SEED)

    recommendation = np.where(
        (within_segment + segment) % 3 == 0,
        "high",
        np.where((within_segment + segment) % 3 == 1, "middle", "low"),
    )

    return pd.DataFrame(
        {
            "exposure_score": within_segment.astype(float),
            "recommendation": recommendation,
            "expectation": segment,
            "age": rng.integers(20, 61, size=len(segment)),
            "region": np.resize(
                np.array(["Kanto", "Kansai", "Kyushu"], dtype=object),
                len(segment),
            ),
        }
    )


def generate_research_sample() -> pd.DataFrame:
    within_segment, segment = _base_data()
    rng = np.random.default_rng(SEED)

    confounders = {
        "SQ1": rng.normal(0, 1, len(segment)),
        "SQ3": rng.normal(0, 1, len(segment)),
        "SQ8": rng.normal(0, 1, len(segment)),
        "Q5_2": rng.integers(1, 6, len(segment)),
        "Q12_2": rng.integers(1, 6, len(segment)),
        "Q14_2": rng.integers(1, 6, len(segment)),
        "Q28": rng.normal(0, 1, len(segment)),
        "Q29": rng.normal(0, 1, len(segment)),
        "Q30": rng.normal(0, 1, len(segment)),
        "Q31": rng.normal(0, 1, len(segment)),
        "Q38": rng.normal(0, 1, len(segment)),
        "Q39": rng.normal(0, 1, len(segment)),
        "Q41": rng.normal(0, 1, len(segment)),
    }

    return pd.DataFrame(
        {
            "Q2_9": within_segment.astype(float),
            "Q4_1": (within_segment % 5) + 1,
            "Q7_4": segment,
            **confounders,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_api_sample().to_csv(
        OUTPUT_DIR / "api_sample.csv",
        index=False,
        encoding="utf-8",
    )
    generate_research_sample().to_csv(
        OUTPUT_DIR / "research_sample.csv",
        index=False,
        encoding="shift-jis",
    )


if __name__ == "__main__":
    main()
