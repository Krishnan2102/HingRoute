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
    train_df=pd.read_csv("data/processed/train.csv")
    val_df=pd.read_csv("data/processed/val.csv")
    test_df=pd.read_csv("data/processed/test.csv")
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
def run_zero_shot_baseline(
    test_df: pd.DataFrame,
    model_name: str = "openai/gpt-oss-20b",
    max_eval_samples: int = 50,  # Cap evaluation sample size to prevent quota hang
) -> Dict[str, Any]:
    print(f"\n--- Running Baseline 2: Zero-Shot LLM ({model_name}) ---")
    valid_intents = set(test_df["label"].unique())
    preds = []

    eval_slice = test_df.head(max_eval_samples) if max_eval_samples else test_df
    total = len(eval_slice)
    print(f"Evaluating {total} samples on {model_name}...")

    start_time = time.perf_counter()
    for idx, (_, row) in enumerate(eval_slice.iterrows()):
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text=row["text"])
        try:
            raw_resp = cached_generate(model=model_name, prompt=prompt, temperature=0.0)
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_resp.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            pred_intent = data.get("intent", "other")
            if pred_intent not in valid_intents:
                pred_intent = "other"
        except Exception as e:
            pred_intent = "other"

        preds.append(pred_intent)

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"Zero-shot progress: {idx + 1}/{total} evaluated...")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / total

    metrics = evaluate_predictions(
        y_true=eval_slice["label"],
        y_pred=preds,
        output_prefix="results/baseline_zeroshot",
    )
    metrics["latency_ms_per_query"] = elapsed_ms
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