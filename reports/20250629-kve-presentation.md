---
date: 7th of July 2025
title: Machine Learning For Profit Based Investing
subtitle: Predictive Modelling (2024 P3A)
institute: HAN - Master Applied Data Science
author:
- Koen van Esterik
- Steven Bontius
---

## Your Role in Today's Meeting

- Your Position: You are the Data Science leadership team of our investment company.
- The Challenge: Our telemarketing division is currently operating at a loss.
- The Proposal: We will present our research findings and a strategic plan for a turnaround.
- Your Objective: Evaluate our proposal and decide whether to approve this project.

---

## Dear stakeholders, we would like to present the findings of our research.

---

# Methodology

1. Exploratory Data Analysis
2. Predictive Modelling
   - Introducing the **Maximum Profit** metric
4. Model Selection
5. Model Evaluation
6. Conclusion
   - Comparison of profit with and without predictive modelling 

---

# Maximum Profit Metric

Introducing the **Maximum Profit** metric to evaluate the performance of models, as well as the profitability of the telemarketing division:

---

$$
\vec{y_{\textit{pred}}} = \sum_{i=1}^{\vec{thresholds}} \begin{cases} 
1 & \text{if } \vec{y_{\textit{probs}}} \geq \textit{threshold}_i \\ 
0 & \text{otherwise} 
\end{cases}
$$

---

$$
\vec{tps}, \vec{fns}, \vec{fps}, \vec{tns} = \sum_{i=1}^{\vec{thresholds}} \text{confusion\_matrix}(\vec{y_{\textit{true}}}, \vec{y_{\textit{pred}}})
$$

---

$$
\textit{total profit} = \max{(\textit{profit per subscription} * \vec{tps} - \textit{cost per call} * \vec{fps})}
$$

---

Basically, we are calculating the profit for each threshold. Then we are plotting these results to create a **Profit Curve**. This plot allows us to find the **Maximum Profit**.

---

# Results

![Model Shortlist](presentation-model-shortlist.png)

---

# Results

![Model Calibration](presentation-model-calibration.png)

---

# Results

![Model Selection](presentation-model-selection.png)

---

# Results

![Learning Curves](presentation-learning-curves.png)

---

# Conclusion

Comparison of profit with and without predictive modelling:

| Procedure                   | Profit  |
| --------------------------- | ------- |
| Call All Prospects          | 10,000  |
| Call Preselected Propspects | 101,100 |

---

# Conclusion

## "Gas op die lolly?"

---

# Remarks

![Instances per Year](number-of-instances-per-year.png)

---

# Remarks

![Proportion of Target Variable per Year](proportion-of-target-variable-per-year.png)

---

# Remarks

![Proportion of Approached Prospects per Year](proportion-of-approached-prospects-per-year.png)

---

## Questions?