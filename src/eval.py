# 1. Imports: numpy for resampling, sklearn for metrics, matplotlib/seaborn for plots

# 2. Function: compute_bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, seed=42)
#    - For i in range(n_bootstraps):
#         - Draw random indices with replacement (same length as dataset)
#         - Compute metric on this resample
#    - Return 2.5th and 97.5th percentiles

# 3. Function: evaluate_predictions(y_true, y_pred, labels=None, output_dir=None)
#    - Compute accuracy
#    - Compute macro-F1
#    - Compute bootstrap 95% CI on macro-F1
#    - Generate per-class classification report
#    - Compute confusion matrix
#    - Save confusion matrix as CSV and heat-map plot (PNG)
#    - Return structured dict of results