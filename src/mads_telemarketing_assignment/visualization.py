from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from mads_telemarketing_assignment.config import (
    HONOLULU_BLUE,
    IMPERIAL_RED,
    PERSIAN_GREEN,
)


def binary_feature_plot(feature: pd.Series, feature_name: str) -> None:
    value_counts = feature.value_counts().sort_index()
    plt.figure(figsize=(6, 6))
    plt.bar(
        value_counts.index.astype(str),
        value_counts.values.tolist(),
        color=[IMPERIAL_RED, PERSIAN_GREEN],
        edgecolor="white",
        width=1.0,
    )
    plt.title(feature_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def categorical_feature_plot(feature: pd.Series, feature_name: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.hist(
        feature,
        bins=len(feature.unique()),
        edgecolor="white",
        color=HONOLULU_BLUE,
    )
    plt.title(feature_name)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def numerical_feature_plot(feature: pd.Series, feature_name: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.hist(
        feature,
        bins=30,
        edgecolor="white",
        color=HONOLULU_BLUE,
    )
    plt.title(feature_name)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_profit_thresholds(
    total_profits: NDArray[np.int32],
    thresholds: NDArray[np.float32],
    optimal_threshold: float,
    ax: Axes,
    model_name: Optional[str] = None,
) -> None:
    """
    Plot total expected profit across classification thresholds.

    Parameters
    ----------
    total_profits : NDArray[np.int32]
        Array of total expected profits for each threshold.

    thresholds : NDArray[np.float32]
        Array of threshold values.

    optimal_threshold : float
        Threshold value that yields the maximum profit.

    ax : Axes
        Matplotlib Axes object to plot on.

    model_name : Optional[str], default=None
        Name of the model for labeling the plot.

    Returns
    -------
    None
    """

    ax.plot(
        thresholds,
        total_profits,
        marker="o",
        color=HONOLULU_BLUE,
        linestyle="-",
    )
    ax.plot(
        thresholds,
        total_profits,
        linestyle="-",
        color=HONOLULU_BLUE,
        linewidth=4,
        label="Total Expected Profit",
    )
    ax.axvline(
        optimal_threshold,
        color=IMPERIAL_RED,
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.2f}",
    )

    if model_name:
        ax.set_title(model_name)
    else:
        ax.set_title("Expected Profit Across Classification Thresholds")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Profit")
    ax.legend()
