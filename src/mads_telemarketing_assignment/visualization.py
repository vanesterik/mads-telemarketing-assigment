import matplotlib.pyplot as plt
import pandas as pd

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
