import json
import os
import re
import pandas as pd
from src.data_prep import prepare_banking77_base, PROMPT_TEMPLATE, clean_text, is_valid_variant
from src.llm_utils import cache, get_cache_key

# 1. Load the exact base English splits (with identical seed=42)
base_df = prepare_banking77_base(seed=42)

model_name = "openai/gpt-oss-20b"  # or whichever model ID you passed to cached_generate
records = []
styles = ["casual", "noisy", "mixed"]

print("Scanning local diskcache for completed rows...")
found_count = 0

for idx, row in base_df.iterrows():
    prompt = PROMPT_TEMPLATE.format(text=row["orig_text"])
    key = get_cache_key(model=model_name, prompt=prompt)
    
    # Check if this exact prompt exists in diskcache
    if key in cache:
        found_count += 1
        raw_resp = cache[key]
        try:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_resp.strip(), flags=re.MULTILINE)
            variants = json.loads(cleaned)
            if not isinstance(variants, list):
                variants = []
        except Exception:
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
                    "split": row["split"]
                })

print(f"Found {found_count} completed base queries in cache!")
print(f"Extracted {len(records)} total Hinglish variant rows.")

# 2. Save progress checkpoint
os.makedirs("data/processed", exist_ok=True)
df_recovered = pd.DataFrame(records)
df_recovered.to_csv("data/processed/generated_variants.csv", index=False)
print("Saved checkpoint to data/processed/generated_variants.csv")

# 3. Verify leak-free splits
train_ids = set(df_recovered[df_recovered["split"] == "train"]["orig_id"])
val_ids = set(df_recovered[df_recovered["split"] == "val"]["orig_id"])
test_ids = set(df_recovered[df_recovered["split"] == "test"]["orig_id"])

assert train_ids.isdisjoint(val_ids), "Data leakage between train and val!"[cite: 10]
assert train_ids.isdisjoint(test_ids), "Data leakage between train and test!"[cite: 10]
assert val_ids.isdisjoint(test_ids), "Data leakage between val and test!"[cite: 10]

# 4. Export interim train/val/test splits so you can proceed with Block 2
for split_name in ["train", "val", "test"]:
    split_df = df_recovered[df_recovered["split"] == split_name]
    out_path = f"data/processed/{split_name}.csv"
    split_df.to_csv(out_path, index=False)
    print(f"Exported {out_path} ({len(split_df)} rows)")

print("\nInterim dataset saved and verified successfully!")