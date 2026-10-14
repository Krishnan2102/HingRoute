#1. Load train.csv, val.csv, and test.csv.
#2. Train Majority Predictor -> evaluate on val and test using evaluate_predictions().
#3. Fit char_wb TF-IDF + LogisticRegression on train -> evaluate on val and test.
#4. Run cached Zero-Shot LLM on test -> evaluate on test.
#5. Record macro-F1, accuracy, latency per query, and estimated cost per 1,000 queries.
#6. Save summary table to results/phase1_results.csv.