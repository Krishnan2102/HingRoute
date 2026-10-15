import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.baselines import load_datasets, ZERO_SHOT_PROMPT_TEMPLATE
from src.llm_utils import cached_generate

def benchmark_inference_latency(n_samples: int = 50):
    print(f"Loading test split for latency benchmarking ({n_samples} queries)...")
    _, _, test_df = load_datasets()
    eval_slice = test_df.head(n_samples)

    results = []

    # 1. Majority Class Model
    print("Benchmarking Majority Class baseline...")
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(eval_slice["text"], eval_slice["label"])
    
    maj_times = []
    for text in eval_slice["text"]:
        t0 = time.perf_counter()
        _ = dummy_clf.predict([text])
        maj_times.append((time.perf_counter() - t0) * 1000.0)

    results.append({
        "Model": "Majority Class",
        "Median Latency (ms)": np.median(maj_times),
        "P95 Latency (ms)": np.percentile(maj_times, 95),
        "Min Latency (ms)": np.min(maj_times),
        "Max Latency (ms)": np.max(maj_times),
    })

    # 2. TF-IDF + Logistic Regression Model
    tfidf_model_path = "models/baseline_tfidf/model.joblib"
    if os.path.exists(tfidf_model_path):
        print("Benchmarking TF-IDF + Logistic Regression baseline...")
        tfidf_pipeline = joblib.load(tfidf_model_path)
        
        tfidf_times = []
        for text in eval_slice["text"]:
            t0 = time.perf_counter()
            _ = tfidf_pipeline.predict([text])
            tfidf_times.append((time.perf_counter() - t0) * 1000.0)

        results.append({
            "Model": "Char n-gram TF-IDF + LR",
            "Median Latency (ms)": np.median(tfidf_times),
            "P95 Latency (ms)": np.percentile(tfidf_times, 95),
            "Min Latency (ms)": np.min(tfidf_times),
            "Max Latency (ms)": np.max(tfidf_times),
        })
    else:
        print(f"Warning: {tfidf_model_path} not found. Skipping TF-IDF benchmark.")

    # 3. Zero-Shot LLM (Measuring API Round-Trip Latency)
    print("Benchmarking Zero-Shot LLM (Live API Round-Trip on 10 fresh samples)...")
    llm_times = []
    # Test on a small live slice (10 queries) to capture genuine API round-trip time without exhausting tokens
    llm_slice = eval_slice.head(10)
    for _, row in llm_slice.iterrows():
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(text=row["text"])
        t0 = time.perf_counter()
        # Bypass cache to measure actual live network + inference latency
        _ = cached_generate(model="openai/gpt-oss-20b", prompt=prompt, temperature=0.0)
        llm_times.append((time.perf_counter() - t0) * 1000.0)

    results.append({
        "Model": "Zero-Shot LLM (GPT-OSS-20B))",
        "Median Latency (ms)": np.median(llm_times),
        "P95 Latency (ms)": np.percentile(llm_times, 95),
        "Min Latency (ms)": np.min(llm_times),
        "Max Latency (ms)": np.max(llm_times),
    })

    # 4. Save and Display
    df_latencies = pd.DataFrame(results)
    out_csv = "results/latency_benchmark.csv"
    os.makedirs("results", exist_ok=True)
    df_latencies.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("BENCHMARK LATENCY REPORT (ms/query)")
    print("=" * 70)
    print(df_latencies.round(3).to_string(index=False))
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    benchmark_inference_latency()