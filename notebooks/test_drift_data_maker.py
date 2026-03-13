import pandas as pd
import numpy as np
import os
import shutil

# ---------------------------------------------------
# Clean previous generated folders
# ---------------------------------------------------

folders_to_clear = ["artifacts/data", "data/production_batches"]

for folder in folders_to_clear:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder}")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

df = pd.read_csv("data/raw/gemstone.csv")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)

# ---------------------------------------------------
# Split dataset
# ---------------------------------------------------

train_end = int(n * 0.7)
test_end = int(n * 0.8)

reference_data = df[:train_end]
test_data = df[train_end:test_end]
production_data = df[test_end:]

# ---------------------------------------------------
# Create folders
# ---------------------------------------------------

os.makedirs("artifacts/data", exist_ok=True)
os.makedirs("data/production_batches", exist_ok=True)

# ---------------------------------------------------
# Save reference + test datasets
# ---------------------------------------------------

reference_data.to_csv("artifacts/data/reference_data.csv", index=False)
test_data.to_csv("artifacts/data/test_data.csv", index=False)

# ---------------------------------------------------
# Create production batches
# ---------------------------------------------------

batch_size = 500
drift_batch_number = 4   # Inject drift here

batch_id = 1

for i in range(0, len(production_data), batch_size):

    batch = production_data[i:i + batch_size].copy()

    # ---------------------------------------------------
    # Inject synthetic drift
    # ---------------------------------------------------

    if batch_id == drift_batch_number:

        print("Injecting synthetic drift in batch", batch_id)

        batch["carat"] = batch["carat"] + np.random.normal(1.5, 0.3, size=len(batch))
        batch["depth"] = batch["depth"] + np.random.normal(8, 2, size=len(batch))

    batch.to_csv(
        f"data/production_batches/batch_{batch_id}.csv",
        index=False
    )

    batch_id += 1

print("Dataset split completed with controlled drift injection.")