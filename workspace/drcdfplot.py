import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import japanize_matplotlib

japanize_matplotlib.japanize()


def drcdf_plot(
    cdf_seg: pd.DataFrame,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> list[Path]:
    """
    DR-CDFの計算結果を描画する。

    Parameters
    ----------
    cdf_seg:
        oc_dr_cdf_by_seg()が返したDataFrame。
    output_dir:
        画像の保存先。Noneの場合は保存しない。
    show:
        Trueの場合のみグラフを画面表示する。

    Returns
    -------
    保存した画像のパス一覧。
    """

    required_columns = {
        "seg",
        "c",
        "threshold",
        "F1_dr",
        "F0_dr",
        "tau_c",
        "ci_low",
        "ci_high",
    }

    missing_columns = required_columns - set(cdf_seg.columns)

    if missing_columns:
        raise ValueError(
            f"DR-CDF描画に必要な列がありません: "
            f"{sorted(missing_columns)}"
        )

    if cdf_seg.empty:
        raise ValueError("DR-CDFの結果が空です。")

    save_directory = None

    if output_dir is not None:
        save_directory = Path(output_dir)
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    saved_paths: list[Path] = []

    for segment_id, segment_data in cdf_seg.groupby("seg"):
        segment_data = segment_data.sort_values("c")

        # ==========================================
        # 処置群・対照群のCDF
        # ==========================================

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            segment_data["threshold"],
            segment_data["F1_dr"],
            marker="o",
            label="処置群",
        )

        ax.plot(
            segment_data["threshold"],
            segment_data["F0_dr"],
            marker="o",
            label="対照群",
        )

        ax.set_xlabel("アウトカムのしきい値")
        ax.set_ylabel("累積確率")
        ax.set_title(f"DR-CDF：セグメント{segment_id}")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_directory is not None:
            path = (
                save_directory
                / f"drcdf_segment_{int(segment_id)}.png"
            )

            fig.savefig(
                path,
                dpi=200,
                bbox_inches="tight",
            )

            saved_paths.append(path)

        if show:
            plt.show()

        plt.close(fig)

        # ==========================================
        # 処置群と対照群のCDF差
        # ==========================================

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            segment_data["threshold"],
            segment_data["tau_c"],
            marker="o",
            label="DR-CDF差",
        )

        ax.fill_between(
            segment_data["threshold"],
            segment_data["ci_low"],
            segment_data["ci_high"],
            alpha=0.3,
            label="95%信頼区間",
        )

        # 効果なしの基準線
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel("アウトカムのしきい値")
        ax.set_ylabel("処置群 − 対照群")
        ax.set_title(f"DR-CDF差：セグメント{segment_id}")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_directory is not None:
            path = (
                save_directory
                / f"drcdf_difference_segment_{int(segment_id)}.png"
            )

            fig.savefig(
                path,
                dpi=200,
                bbox_inches="tight",
            )

            saved_paths.append(path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths