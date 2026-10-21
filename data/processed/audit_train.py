import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
df = pd.read_csv("generated_variants.csv")
print(f"Initial shape: {df.shape}")

# ---------------------------------------------------------
# 2. Programmatic Deduplication Across Splits
# ---------------------------------------------------------
# If the exact same normalized text appears in multiple splits,
# prioritize keeping train, and remove it from eval sets
# to prevent evaluation leakage.

df["text_clean"] = df["text"].astype(str).str.strip().str.lower()

# Identify cross-split duplicates
dupes_mask = df.duplicated(subset=["text_clean"], keep=False)
dupes = df[dupes_mask]

split_priority = {"train": 0, "val": 1, "test": 2}
df["split_rank"] = df["split"].map(split_priority)

# Sort by split rank so 'train' comes first, then drop duplicates on text_clean
before_count = len(df)
df = df.sort_values(by=["split_rank", "orig_id"]).drop_duplicates(
    subset=["text_clean"], keep="first"
).reset_index(drop=True)
df.drop(columns=["split_rank"], inplace=True)

dropped_cross_dupes = before_count - len(df)
print(f"Dropped {dropped_cross_dupes} duplicate text instances across splits.")
print(f"Current split counts:\n{df['split'].value_counts()}\n")

# ---------------------------------------------------------
# 3. Grouped Out-Of-Fold (OOF) Audit on Train
# ---------------------------------------------------------
train_df = df[df["split"] == "train"].copy().reset_index(drop=True)

# Map labels to integers
labels = sorted(train_df["label"].unique())
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for lbl, i in label2id.items()}

train_df["label_id"] = train_df["label"].map(label2id)

# 5-fold GroupKFold by orig_id to ensure no variant leaks into validation
gkf = GroupKFold(n_splits=5)
groups = train_df["orig_id"].values
X = train_df["text_clean"].values
y = train_df["label_id"].values

oof_probs = np.zeros((len(train_df), len(labels)))

print("Running 5-fold GroupKFold OOF audit on train...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0, random_state=42)
    )
    
    pipeline.fit(X[train_idx], y[train_idx])
    oof_probs[val_idx] = pipeline.predict_proba(X[val_idx])

# Probability assigned to the ground-truth annotated label
true_class_probs = oof_probs[np.arange(len(train_df)), y]
train_df["prob_given_label"] = true_class_probs
train_df["pred_label"] = [id2label[i] for i in oof_probs.argmax(axis=1)]
train_df["pred_confidence"] = oof_probs.max(axis=1)

# ---------------------------------------------------------
# 4. Flag Suspicious Rows & Group by orig_id
# ---------------------------------------------------------
# Threshold: Flag where model gives ground-truth label < 15% probability
SUSPICIOUS_THRESHOLD = 0.15
suspicious_mask = train_df["prob_given_label"] < SUSPICIOUS_THRESHOLD
flagged_rows = train_df[suspicious_mask].copy()

print(f"Total train rows flagged (p < {SUSPICIOUS_THRESHOLD}): {len(flagged_rows)} ({len(flagged_rows)/len(train_df):.2%})")
print("\nFlagged breakdown by ground truth label:")
print(flagged_rows["label"].value_counts())

# Group flagged cases by orig_id so you can see all variants of the parent query
flagged_orig_ids = flagged_rows["orig_id"].unique()
audit_view = train_df[train_df["orig_id"].isin(flagged_orig_ids)].sort_values(
    by=["orig_id", "prob_given_label"]
)

export_cols = [
    "orig_id",
    "id",
    "label",
    "pred_label",
    "prob_given_label",
    "pred_confidence",
    "style",
    "text"
]

audit_view[export_cols].to_csv("to_review_train_oof.csv", index=False)
print(f"\nSaved {len(audit_view)} variant rows across {len(flagged_orig_ids)} parent orig_ids to 'to_review_train_oof.csv'.")