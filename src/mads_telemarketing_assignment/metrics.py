from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


def calculate_profit_thresholds(
    y_true: NDArray[np.int32],
    y_probs: NDArray[np.float32],
    revenue_per_success: int = 10,
    cost_per_call: int = 1,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    float,
    int,
    float,
]:
    """
    Compute the total profit for each threshold, identify the maximum profit and
    corresponding threshold, and calculate precision and recall at the optimal threshold.

    Parameters
    ----------
    y_true : NDArray[np.int32]
        Array of true binary labels.

    y_probs : NDArray[np.float32]
        Array of predicted probabilities for the positive class.

    revenue_per_success : int, optional
        Revenue assigned to a true positive (default: 10).

    cost_per_call : int, optional
        Cost assigned to each true positive and false positive (default: 1).

    Returns
    -------
    thresholds : NDArray[np.float32]
        Array of threshold values.

    profits : NDArray[np.int32]
        Array with profit for each threshold.

    optimal_threshold : float
        The threshold corresponding to the maximum profit.

    profit : int
        The maximum profit across all thresholds.

    profit_margin : float
        The profit margin calculated by the maximum profit divided by the revenue.
    """

    # Define thresholds from 0.0 to 1.0 with a step of 0.01
    thresholds = np.arange(0.0, 1, 0.01)

    # Initialize arrays to store counts of true positives, false negatives,
    # false positives, and true negatives for each threshold
    tps = np.zeros(len(thresholds), dtype=np.int32)
    fns = np.zeros(len(thresholds), dtype=np.int32)
    fps = np.zeros(len(thresholds), dtype=np.int32)
    tns = np.zeros(len(thresholds), dtype=np.int32)

    for i, threshold in enumerate(thresholds):
        # Apply threshold to get predicted labels
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Store the counts for each threshold
        tps[i] = tp
        fns[i] = fn
        fps[i] = fp
        tns[i] = tn

    # Calculate the total profit for each threshold
    revenues = revenue_per_success * tps
    profits = revenues - cost_per_call * (fps + tps)

    # Calculate the total profit for each threshold and identify the threshold
    # with the maximum profit
    profit_index = np.argmax(profits)
    profit = profits[profit_index]
    revenue = revenues[profit_index]
    profit_margin = profit / revenue if revenue > 0 else 0.0
    optimal_threshold = thresholds[profit_index]

    return (
        thresholds,
        profits,
        float(optimal_threshold),
        int(profit),
        float(profit_margin),
    )


def calculate_cost_estimates(
    X: pd.DataFrame,
    y: NDArray[np.int32],
    preparation_time: int = 3,  # Preparation is estimated to be 3 minutes per call
    hourly_wage: int = 35,
    revenue_per_success: int = 200,
) -> Tuple[
    float,
    float,
    float,
    int,
    int,
    int,
]:
    """
    Calculate the cost per call, total costs, profit, and revenue based on the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the telemarketing data with columns 'campaign' and 'duration'.

    y : NDArray[np.int32]
        Array of binary labels indicating success (1) or failure (0) of the call.

    preparation_time : int, optional
        Preparation time in minutes per call (default: 3).

    hourly_wage : int, optional
        The hourly wage used in the calculations (default: 35).

    revenue_per_success : int, optional
        The revenue per successful call (default: 200).

    Returns
    -------
    cost_per_call : float
        The cost per call calculated based on the average call duration and preparation time.

    total_costs : float
        The total costs calculated based on the number of calls and cost per call.

    profit : float
        The profit calculated as revenue minus total costs.

    revenue : int
        The total revenue calculated based on the number of successful calls.

    hourly_wage : int, optional
        The hourly wage used in the calculations (default: 35).

    revenue_per_success : int, optional
        The revenue per successful call (default: 200).

    """
    mean_calls = X["campaign"].mean()
    mean_duration = X["duration"].mean()
    mean_duration_minutes = mean_duration / 60
    mean_call_per_customer = mean_duration_minutes * mean_calls

    total_prep_time = preparation_time * mean_calls
    mean_time_per_customer = mean_call_per_customer + total_prep_time
    cost_per_call = hourly_wage * (mean_time_per_customer / 60)

    total_costs = len(X) * cost_per_call
    revenue = len(y[y == 1]) * revenue_per_success
    profit = revenue - total_costs

    return (
        cost_per_call,
        total_costs,
        profit,
        revenue,
        hourly_wage,
        revenue_per_success,
    )
