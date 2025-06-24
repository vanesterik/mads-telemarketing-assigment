from typing import List, Union

import pandas as pd


def print_unique_values(df: pd.DataFrame, features: Union[str, List[str]]) -> None:
    """
    Print unique values for each passed feature in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features.

    features : Union[str, List[str]]
        Feature name or list of feature names to report unique values for.

    Returns
    -------
    None

    """

    if isinstance(features, str):
        features = [features]

    for feature in features:

        if feature not in df.columns:
            continue

        print(
            df[feature]
            .value_counts(dropna=False, normalize=True)
            .sort_index()
            .to_string()
        )
        print(f"Unique values: {df[feature].nunique()}\n")
