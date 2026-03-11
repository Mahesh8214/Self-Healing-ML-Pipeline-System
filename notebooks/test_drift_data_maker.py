import pandas as pd
import os
import shutil  # Add this import

# Delete previous folders (safe, ignores if missing)
folders_to_clear = ["artifacts/data", "data/production_batches"]
for folder in folders_to_clear:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder}")

# Load dataset
df = pd.read_csv("data/raw/gemstone.csv")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)

# Split sizes
train_end = int(n * 0.7)
test_end = int(n * 0.8)

reference_data = df[:train_end]
test_data = df[train_end:test_end]
production_data = df[test_end:]

# Create fresh folders
os.makedirs("artifacts/data", exist_ok=True)
os.makedirs("data/production_batches", exist_ok=True)

# Save reference and test
reference_data.to_csv("artifacts/data/reference_data.csv", index=False)
test_data.to_csv("artifacts/data/test_data.csv", index=False)

# Create production batches
batch_size = 300

for i in range(0, len(production_data), batch_size):
    batch = production_data[i:i+batch_size]
    batch_id = i // batch_size + 1
    batch.to_csv(f"data/production_batches/batch_{batch_id}.csv", index=False)

print("Dataset split completed fresh.")
