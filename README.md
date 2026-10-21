# Hinglish Banking Intent Classifier: Architectural Shootout

A machine learning pipeline that classifies noisy, transliterated Hinglish banking queries into 12 customer-service intents. The project benchmarks a lightweight linear baseline against zero-shot LLM prompting and fine-tuned multilingual transformers, to find the best trade-off between accuracy, compute cost, and inference latency.

## Project Objective

Do large multilingual transformers give enough of a performance edge on code-switched (Hindi-English) text classification to justify their computational requirements and GPU dependency, compared with a character-level classical approach?

## Dataset

Synthetic banking queries written in Romanized Hinglish (for example *"mera card block ho gaya hai"*, *"nayi checkbook kaise mangwau"*), mapped to 12 intents such as `card_arrival`, `lost_or_stolen_card`, and `compromised_card`. The text is informal, with inconsistent and phonetic spellings.

- Splits: `train.csv`, `val.csv`, `test.csv` in `data/processed/`
- Metric: macro-F1 on the held-out test split

## The Architectural Shootout

Three paradigms were evaluated:

1. **Lightweight baseline:** character n-gram TF-IDF + Logistic Regression. Character windows capture phonetic spelling variation at the sub-word level.
2. **Zero-shot LLM:** a general-purpose LLM called through an API, with no task-specific training.
3. **Transformer fine-tuning:**
   - **XLM-RoBERTa** (`xlm-roberta-base`): multilingual transformer pre-trained on 100 languages.
   - **MuRIL** (`google/muril-base-cased`): transformer pre-trained on Indian languages, including transliterated text.

## Results

| Model | Test Macro-F1 | Inference Latency | Compute | Note |
| --- | --- | --- | --- | --- |
| **Char n-gram TF-IDF + LR** | 0.922 | **~0.17 ms/query** | CPU | **Selected for production** |
| XLM-RoBERTa (fine-tuned) | **0.929** | _TBD (add measured value)_ | GPU | +0.7 pt macro-F1 over TF-IDF |
| Zero-shot LLM | 0.916 | ~2879 ms/query | API / network | High latency and per-call cost |
| MuRIL (fine-tuned) | 0.643 | _TBD (add measured value)_ | GPU | Did not converge in this setup |

Latency measurements are in `results/latency_benchmark.csv`.

## Decision and Conclusion

**Selected model: character n-gram TF-IDF + Logistic Regression.**

XLM-RoBERTa has the highest macro-F1 (0.929 vs 0.922), but the 0.7-point gain does not justify its costs: it needs GPU infrastructure, adds substantial latency, and is a much larger model to deploy and maintain. The TF-IDF pipeline runs in well under a millisecond per query on a standard CPU, and its character n-grams handle Hinglish spelling variation well.

MuRIL, although pre-trained on Indian languages, reached only 0.643 macro-F1 and is treated as a failed run in this setup (see Limitations).

On this dataset, an optimized statistical model delivers nearly the same accuracy as a fine-tuned transformer at a tiny fraction of the cost and latency.

## Limitations

- **Synthetic data.** The dataset is synthetic, so scores are likely optimistic. Results on real customer messages may differ, and the ranking of models could change. A human-written held-out test set would be the next step.
- **The 0.7-point gap may be noise.** Results come from a single split without confidence intervals or multiple seeds. Bootstrap CIs on the test set would show whether XLM-R and TF-IDF are statistically distinguishable.
- **MuRIL's failure is unexplained.** A score this low, from a model pre-trained on transliterated Indian-language text, more likely points to a training-configuration issue (learning rate, epochs, label mapping, tokenization) than to a limit of the model. It should not be read as evidence that MuRIL cannot handle this domain until it is debugged.
- **Transformer latency not yet reported.** Add measured ms/query for both transformers (and state the hardware) so the latency comparison is complete.
- **Single domain.** Findings apply to this 12-intent banking task and may not generalize to other code-switched tasks.

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Original unformatted query data
│   └── processed/            # train.csv, val.csv, test.csv
├── notebooks/
│   ├── phase1_baselines.ipynb               # TF-IDF and zero-shot LLM experiments
│   └── phase2_transformer_finetuning.ipynb  # XLM-RoBERTa and MuRIL training
└── results/
    ├── latency_benchmark.csv  # Inference latency profiling
    ├── phase1_results.csv     # Baseline metric logs
    └── phase2_results.csv     # Transformer evaluation metrics
```

## Reproducing the Experiments

1. Run `notebooks/phase1_baselines.ipynb` for the TF-IDF and zero-shot LLM results.
2. Run `notebooks/phase2_transformer_finetuning.ipynb` (GPU recommended) for the XLM-RoBERTa and MuRIL results.
3. Compare outputs against the files in `results/`.
