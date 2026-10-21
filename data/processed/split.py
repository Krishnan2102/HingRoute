import pandas as pd

# Load your manually cleaned dataset
df = pd.read_csv("generated_variants.csv")

# Filter and save the new, clean splits
df[df['split'] == 'train'].to_csv("train_clean.csv", index=False)
df[df['split'] == 'val'].to_csv("val_clean.csv", index=False)
df[df['split'] == 'test'].to_csv("test_clean.csv", index=False)

print(f"Saved train_clean.csv: {len(df[df['split'] == 'train'])} rows")
print(f"Saved val_clean.csv: {len(df[df['split'] == 'val'])} rows")
print(f"Saved test_clean.csv: {len(df[df['split'] == 'test'])} rows")