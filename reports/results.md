# Results

16 athletes, 1713 athlete-days with a next-day target.

## Same athletes, unseen future days (temporal 70/30 split)

| Model                      |   MAE |   RMSE |   Within ±1 point |   Low-readiness AUC |
|:---------------------------|------:|-------:|------------------:|--------------------:|
| Baseline: same as today    | 1.106 |  1.626 |             0.736 |               0.597 |
| Baseline: personal average | 1.27  |  1.657 |             0.549 |               0.419 |
| Ridge regression           | 0.982 |  1.321 |             0.611 |               0.592 |
| Gradient boosting          | 1.04  |  1.394 |             0.59  |               0.603 |

## New athletes (leave-one-athlete-out)

| Model                   |   MAE |   RMSE |   Within ±1 point |
|:------------------------|------:|-------:|------------------:|
| Baseline: same as today | 1.149 |  1.688 |             0.696 |
| Ridge regression        | 1.067 |  1.404 |             0.555 |
| Gradient boosting       | 1.236 |  1.656 |             0.532 |

## Low-readiness day classifier (temporal 70/30 split)

A low day is at or below the athlete's own mean minus 1 SD (training period). Low-day rate: 15.0% (train), 17.6% (test, 432 days).

| Model                               |   ROC-AUC |   PR-AUC |   Recall @20% flagged |   Precision @20% flagged |   Brier score |
|:------------------------------------|----------:|---------:|----------------------:|-------------------------:|--------------:|
| Baseline: low readiness today       |     0.597 |    0.242 |                 0.303 |                    0.267 |       nan     |
| Baseline: below personal norm today |     0.593 |    0.295 |                 0.329 |                    0.291 |       nan     |
| Logistic regression                 |     0.562 |    0.221 |                 0.276 |                    0.244 |         0.217 |
| Gradient boosting                   |     0.677 |    0.309 |                 0.434 |                    0.384 |         0.184 |

Athletes with load features masked (fewer than 5 logged sessions): p03, p09, p16.

## Top drivers (mean |SHAP|, gradient boosting)

|                            |   Mean |SHAP| |
|:---------------------------|--------------:|
| Readiness, 7-day mean      |         0.563 |
| Readiness today            |         0.214 |
| Training load today (sRPE) |         0.19  |
| Resting heart rate         |         0.132 |
| Chronic load (28 d)        |         0.105 |
| Fitbit sleep score         |         0.079 |
| Sleep score vs baseline    |         0.072 |
| Deep sleep (min)           |         0.065 |
| Training monotony          |         0.063 |
| Training strain            |         0.062 |
| Acute load (7 d)           |         0.061 |
| Stress today               |         0.06  |
