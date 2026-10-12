import json
import os
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.llm_utils import cached_generate

CHOSEN_INTENTS = [
    "lost_or_stolen_card",
    "compromised_card",
    "card_not_working",
    "card_arrival",
    "transfer_not_received_by_recipient",
    "cancel_transfer",
    "transfer_fee_charged",
    "pending_card_payment",
    "declined_card_payment",
    "extra_charge_on_statement",
    "top_up_failed",
]

PROMPT_TEMPLATE = """You are helping build a test set of customer messages sent to an Indian bank's support chat.
Rewrite the English message below as 3 different messages a real customer might type:
1. Romanized Hinglish, casual
2. Romanized Hinglish with typos, missing vowels, SMS-style shortcuts
3. Mostly English with a few Hindi words mixed in
Keep exactly the same meaning and problem. Do not name the category. Do not add details.
Return ONLY a JSON list of 3 strings.

English message: "{text}"
"""


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unnecessary enclosing quotes."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip('"\'')


def is_valid_variant(variant: str, original: str, label: str) -> bool:
    """Validate variant: not empty, <= 200 chars, not identical, no leaked label."""
    if not variant or len(variant) > 200:
        return False
    if variant.lower() == original.lower():
        return False
    # Avoid accidental leakage of intent name in generated text
    if label.replace("_", " ") in variant.lower():
        return False
    return True


def prepare_banking77_base(seed: int = 42) -> pd.DataFrame:
    print("Downloading raw Banking77 CSVs directly via pandas...")
    
    # Fetch raw CSVs bypassing the datasets library entirely
    train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    
    # Combine train + test splits to create our own clean, leak-free split
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # The raw CSV uses 'category' instead of integer labels
    full_df = full_df.rename(columns={"category": "label_name"})

    # 1. Filter chosen intents
    selected = full_df[full_df["label_name"].isin(CHOSEN_INTENTS)].copy()

    # 2. Build 'other' class (~10% size of chosen set) from the remaining pool
    target_other_count = int(len(selected) * 0.10)
    other_pool = full_df[~full_df["label_name"].isin(CHOSEN_INTENTS)]
    other_sample = other_pool.sample(n=target_other_count, random_state=seed).copy()
    other_sample["label_name"] = "other"

    combined = pd.concat([selected, other_sample], ignore_index=True)
    combined = combined.rename(columns={"text": "orig_text", "label_name": "label"})
    combined["orig_id"] = combined.index.astype(str)

    # 3. Split first (70% train, 15% val, 15% test) stratified by label
    tr, tmp = train_test_split(
        combined.index,
        test_size=0.30,
        stratify=combined["label"],
        random_state=seed,
    )
    va, te = train_test_split(
        tmp,
        test_size=0.50,
        stratify=combined.loc[tmp, "label"],
        random_state=seed,
    )

    combined["split"] = "train"
    combined.loc[va, "split"] = "val"
    combined.loc[te, "split"] = "test"

    print(f"Base dataset prepared: {len(combined)} rows across 12 intents.")
    return combined[["orig_id", "orig_text", "label", "split"]]

def generate_hinglish_variants(df_base: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    subset = df_base.head(limit) if limit else df_base
    print(f"Generating variants for {len(subset)} records...")
    records = []
    styles = ["casual", "noisy", "mixed"]

    for idx, (_, row) in enumerate(subset.iterrows()):
        prompt = PROMPT_TEMPLATE.format(text=row["orig_text"])
        try:
            raw_resp = cached_generate(prompt=prompt)
            # Strip potential Markdown blocks
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_resp.strip(), flags=re.MULTILINE)
            variants = json.loads(cleaned)
            if not isinstance(variants, list):
                variants = []
        except Exception as e:
            variants = []

        for i, var_text in enumerate(variants[:3]):
            var_text = clean_text(var_text)
            if is_valid_variant(var_text, row["orig_text"], row["label"]):
                records.append({
                    "id": f"{row['orig_id']}_{i}",
                    "orig_id": row["orig_id"],
                    "text": var_text,
                    "label": row["label"],
                    "style": styles[i] if i < len(styles) else "other",
                    "split": row["split"],
                })

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(subset)} base rows completed.")

    return pd.DataFrame(records)


def main():
    os.makedirs("data/processed", exist_ok=True)
    base_df = prepare_banking77_base()

    # Generate full dataset
    df_variants = generate_hinglish_variants(base_df)

    # Sanity checks: Verify NO leakage of orig_id across splits
    train_ids = set(df_variants[df_variants["split"] == "train"]["orig_id"])
    val_ids = set(df_variants[df_variants["split"] == "val"]["orig_id"])
    test_ids = set(df_variants[df_variants["split"] == "test"]["orig_id"])

    assert train_ids.isdisjoint(val_ids), "Leakage detected: train & val share orig_ids!"
    assert train_ids.isdisjoint(test_ids), "Leakage detected: train & test share orig_ids!"
    assert val_ids.isdisjoint(test_ids), "Leakage detected: val & test share orig_ids!"

    # Save to data/processed/
    for split_name in ["train", "val", "test"]:
        split_df = df_variants[df_variants["split"] == split_name]
        out_path = f"data/processed/{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(split_df)} rows)")

    print("\nData generation complete and verified!")


if __name__ == "__main__":
    main()