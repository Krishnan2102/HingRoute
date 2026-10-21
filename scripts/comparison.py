import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load clean splits and predictions
# 1. Load clean splits and predictions
train_df = pd.read_csv("data/processed/train_clean.csv")
test_df = pd.read_csv("data/processed/test_clean.csv")
xlmr_df = pd.read_csv("scripts/xlmr_clean_predictions.csv")
muril_df = pd.read_csv("scripts/muril_clean_predictions.csv")

# 2. Fit TF-IDF on train and predict on test
pipeline = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=10.0, random_state=42)
)
pipeline.fit(train_df["text"], train_df["label"])
tfidf_preds = pipeline.predict(test_df["text"])

# 3. Align true labels and all predictions
unique_labels = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}

y_true = np.array([label2id[l] for l in test_df["label"]])
y_pred_tfidf = np.array([label2id[l] for l in tfidf_preds])
y_pred_xlmr = xlmr_df["xlmr_pred"].to_numpy()
y_pred_muril = muril_df["muril_pred"].to_numpy()
cluster_ids = test_df["orig_id"].to_numpy()

# 4. Paired Cluster Bootstrap
def paired_cluster_bootstrap(y_true, y_pred_base, y_pred_test, cluster_ids, n_bootstraps=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    
    diffs = []
    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        
        f1_base = f1_score(y_true[sampled_indices], y_pred_base[sampled_indices], average="macro")
        f1_test = f1_score(y_true[sampled_indices], y_pred_test[sampled_indices], average="macro")
        diffs.append(f1_base - f1_test)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0)
    
    return np.mean(diffs), (ci_lower, ci_upper), p_value

# 5. Evaluate both transformers against TF-IDF
print("\n" + "="*60)
print("PAIRED CLUSTER BOOTSTRAP RESULTS (TF-IDF vs Transformers)")
print("="*60)
for name, y_pred in [("XLM-RoBERTa", y_pred_xlmr), ("MuRIL", y_pred_muril)]:
    mean_diff, (ci_low, ci_high), p_val = paired_cluster_bootstrap(y_true, y_pred_tfidf, y_pred, cluster_ids)
    print(f"TF-IDF vs {name}:")
    print(f"  Mean Difference: {mean_diff:+.4f}")
    print(f"  95% CI:          [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  P-value:         {p_val:.4f}\n")