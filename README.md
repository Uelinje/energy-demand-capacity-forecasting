# Energy Demand & Capacity Forecasting Analytics

## Executive Overview

This project develops a structured short-term energy demand forecasting framework
using historical PJM hourly load data.

The objective is not merely predictive accuracy, but operational relevance —
specifically supporting short-term grid capacity planning and peak demand risk assessment.

The modeling pipeline incorporates:

- Lag-based autoregressive features
- Rolling statistical indicators
- Explicit calendar encoding
- Cyclical seasonality transformation
- Chronological (leakage-safe) validation
- Peak-hour risk evaluation

---

## Business Framing

Grid operators must ensure sufficient capacity to meet demand,
particularly during high-load stress periods.

Forecasting models must therefore:

- Maintain strong overall accuracy
- Remain reliable during peak demand periods
- Minimize dangerous underprediction risk

This project evaluates performance from that operational perspective.

---

## Dataset

- Source: PJM Hourly Energy Load Dataset
- Time Span: ~3.75 years
- Frequency: Hourly
- Total Observations: ~32,700 (after feature engineering)

---

## Modeling Approach

### 1. Baseline (Persistence Model)
Predict next-hour load using previous-hour load.

### 2. Linear Regression
Structured regression using engineered lag, rolling, and seasonal features.

### 3. Random Forest Regressor
Nonlinear ensemble model to capture interaction effects and dynamic load patterns.

---

## Performance Summary

| Model | MAE | RMSE |
|--------|------|-------|
| Baseline | ~1113 MW | ~1476 MW |
| Linear Regression | ~721 MW | ~955 MW |
| Random Forest | ~325 MW | ~471 MW |

Random Forest reduces forecasting error by approximately 70% compared to baseline.

---

## Peak Demand Evaluation

Peak hours defined as top 10% of demand in test set.

- Peak MAE: ~555 MW
- Peak RMSE: ~783 MW
- Underprediction rate during peaks: ~57%
- Relative error during peaks: ~1–2%

Operational Insight:
While underprediction occurs slightly more than half the time during peaks,
the magnitude of error remains limited, indicating stable performance
under high-load conditions.

---

## Key Drivers of Demand

Feature importance analysis indicates:

1. Previous-hour load (lag_1) dominates prediction.
2. Intraday seasonal encoding meaningfully contributes.
3. Rolling volatility provides minor but useful signal.

This confirms strong autoregressive structure in short-term grid demand.

---

## Repository Structure

