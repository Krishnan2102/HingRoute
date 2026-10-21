import os
import time
import json
import re
from typing import Dict,Any,List,Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

from src.eval import evaluate_predictions
from src.llm_utils import cached_generate

RESULTS_PATH ="results/phase1_results.csv"

ZERO_SHOT_PROMPT_TEMPLATE = """You are an intent classifier for an Indian bank's customer support.
Classify the following customer support message into EXACTLY one of these categories:
- lost_or_stolen_card
- compromised_card
- card_not_working
- card_arrival
- transfer_not_received_by_recipient
- cancel_transfer
- transfer_fee_charged
- pending_card_payment
- declined_card_payment
- extra_charge_on_statement
- top_up_failed
- other (use this if the issue doesn't match any of the above categories)

Return ONLY valid JSON matching this schema:
{{"intent": "<category_name>"}}

Customer message: "{text}"
"""

#1. Load train.csv, val.csv, and test.csv.
def load_datasets() ->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """ Loads processed splits from data/processes/."""
    train_df=pd.read_csv("data/processed/train_clean.csv")
    val_df=pd.read_csv("data/processed/val_clean.csv")
    test_df=pd.read_csv("data/processed/test_clean.csv")
    return train_df,val_df,test_df
#2. Train Majority Predictor -> evaluate on val and test using evaluate_predictions().
def run_majority_baseline(train_df:pd.DataFrame,test_df:pd.DataFrame)->Dict[str,Any]:
    """ Baseline 0: Always predicts the most frequent training class."""
    print("\n---Running Baseline 0 : Majority Class---")
    start_time=time.perf_counter()
    clf=DummyClassifier(strategy="most_frequent")
    clf.fit(train_df["text"],train_df["label"])

    preds=clf.predict(test_df["text"])
    elapsed_ms=(time.perf_counter()-start_time)*1000.0/len(test_df)

    metrics=evaluate_predictions(
        y_true=test_df["label"],
        y_pred=preds,
        output_prefix="results/baseline_majority"
        )
    metrics["latency_ms_per_query"]=elapsed_ms
    metrics["cost_per_1k"]=0.0
    return metrics
#3. Fit char_wb TF-IDF + LogisticRegression on train -> evaluate on val and test.
def run_tfidf_baseline(train_df:pd.DataFrame,test_df:pd.DataFrame)->Dict[str,Any]:
    """Baseline 1: Character n-gram TF-IDF +Logistic Regression."""
    print("\n---Running Baseline 1 : Char n-gram TF-IDF + Logistic Regression---")

    pipeline=make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2,5),
            lowercase=True,
            sublinear_tf=True
        ),
        LogisticRegression(max_iter=2000, C=10.0, random_state=42)
    )
    pipeline.fit(train_df["text"],train_df["label"])

    start_time=time.perf_counter()
    preds=pipeline.predict(test_df["text"])
    elapsed_ms= (time.perf_counter() -start_time)*1000.0/len(test_df)

    metrics=evaluate_predictions(
        y_true=test_df["label"],
        y_pred=preds,
        output_prefix="results/baseline_tfidf"
    )
    metrics["latency_ms_per_query"]=elapsed_ms
    metrics["cost_per_1k"]=0.0
    return metrics
#4. Run cached Zero-Shot LLM on test -> evaluate on test.
PRED_CHECKPOINT_FILE = "results/zeroshot_predictions.csv"


def run_zero_shot_baseline(
    test_df: pd.DataFrame,
    model_name: str = "openai/gpt-oss-20b",
) -> Dict[str, Any]:
    print(f"\n--- Running Baseline 2: Zero-Shot LLM ({model_name}) ---")
    os.makedirs("results", exist_ok=True)
    valid_intents = set(test_df["label"].unique())

    # 1. Load previously completed predictions if checkpoint exists
    completed_ids = set()
    if os.path.exists(PRED_CHECKPOINT_FILE):
        existing_df = pd.read_csv(PRED_CHECKPOINT_FILE, dtype={"id": str})
        completed_ids = set(existing_df["id"].dropna().tolist())

    total_samples = len(test_df)
    print(f"Resuming Zero-Shot: {len(completed_ids)}/{total_samples} already evaluated.")

    # 2. Iterate and log progress
    start_time = time.perf_counter()
    newly_processed = 0

    for idx, (_, row) in enumerate(test_df.iterrows(), 1):
        row_id = str(row.get("id", idx))
        if row_id in completed_ids:
            continue

        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text=row["text"])
        try:
            raw_resp = cached_generate(model=model_name, prompt=prompt, temperature=0.0)
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_resp.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            pred_intent = data.get("intent", "other")
            if pred_intent not in valid_intents:
                pred_intent = "other"
        except Exception:
            pred_intent = "other"

        # Append immediately to checkpoint file
        record_df = pd.DataFrame([{
            "id": row_id,
            "text": row["text"],
            "true_label": row["label"],
            "pred_label": pred_intent
        }])
        write_header = not os.path.exists(PRED_CHECKPOINT_FILE)
        record_df.to_csv(PRED_CHECKPOINT_FILE, mode="a", header=write_header, index=False)
        completed_ids.add(row_id)
        newly_processed += 1

        # Real-time progress logger
        pct = (len(completed_ids) / total_samples) * 100
        print(f"[{len(completed_ids)}/{total_samples}] ({pct:.1f}%) | ID: {row_id} -> {pred_intent}")

    elapsed_total = time.perf_counter() - start_time
    avg_latency = (elapsed_total * 1000.0 / newly_processed) if newly_processed > 0 else 0.0

    # 3. Read complete checkpoint to compute metrics
    full_preds_df = pd.read_csv(PRED_CHECKPOINT_FILE, dtype={"id": str})
    
    # Ensure order aligns with test_df if computing metrics
    merged = test_df[["id", "label"]].astype(str).merge(
        full_preds_df[["id", "pred_label"]].astype(str),
        on="id",
        how="inner"
    )

    metrics = evaluate_predictions(
        y_true=merged["label"],
        y_pred=merged["pred_label"],
        output_prefix="results/baseline_zeroshot"
    )
    metrics["latency_ms_per_query"] = avg_latency
    metrics["cost_per_1k"] = 0.02
    return metrics
#5. Record macro-F1, accuracy, latency per query, and estimated cost per 1,000 queries.
#6. Save summary table to results/phase1_results.csv.
def main():
    os.makedirs("results", exist_ok=True)
    train_df, val_df, test_df = load_datasets()
    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 1. Run Baselines
    majority_res = run_majority_baseline(train_df, test_df)
    tfidf_res = run_tfidf_baseline(train_df, test_df)
    zeroshot_res = run_zero_shot_baseline(test_df)

    # 2. Compile Results Table
    summary_rows = [
        {
            "Model": "Majority Class",
            "Accuracy": majority_res["accuracy"],
            "Macro-F1": majority_res["macro_f1"],
            "Macro-F1 95% CI": f"[{majority_res['macro_f1_ci_95'][0]:.3f}, {majority_res['macro_f1_ci_95'][1]:.3f}]",
            "Latency (ms/query)": f"{majority_res['latency_ms_per_query']:.2f}",
            "Cost/1k Queries ($)": majority_res["cost_per_1k"]
        },
        {
            "Model": "TF-IDF (Char n-grams) + LR",
            "Accuracy": tfidf_res["accuracy"],
            "Macro-F1": tfidf_res["macro_f1"],
            "Macro-F1 95% CI": f"[{tfidf_res['macro_f1_ci_95'][0]:.3f}, {tfidf_res['macro_f1_ci_95'][1]:.3f}]",
            "Latency (ms/query)": f"{tfidf_res['latency_ms_per_query']:.2f}",
            "Cost/1k Queries ($)": tfidf_res["cost_per_1k"]
        },
        {
            "Model": "Zero-Shot LLM",
            "Accuracy": zeroshot_res["accuracy"],
            "Macro-F1": zeroshot_res["macro_f1"],
            "Macro-F1 95% CI": f"[{zeroshot_res['macro_f1_ci_95'][0]:.3f}, {zeroshot_res['macro_f1_ci_95'][1]:.3f}]",
            "Latency (ms/query)": f"{zeroshot_res['latency_ms_per_query']:.2f}",
            "Cost/1k Queries ($)": zeroshot_res["cost_per_1k"]
        }
    ]

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_PATH, index=False)
    print("\n" + "="*80)
    print("PHASE 1 BASELINE RESULTS:")
    print("="*80)
    print(df_summary.to_string(index=False))
    print(f"\nSaved results summary to {RESULTS_PATH}")


if __name__ == "__main__":
    main()