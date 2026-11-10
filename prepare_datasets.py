
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Load the original IMDb dataset from A1
dataset = load_dataset("stanfordnlp/imdb")

# Dataset Version 1: original training data
v1 = dataset["train"].to_pandas()

# Dataset Version 2: simulate a change in class distribution
positive = v1[v1["label"] == 1].sample(
    n=5000, random_state=42
)
negative = v1[v1["label"] == 0].sample(
    n=10000, random_state=42
)

v2 = pd.concat([positive, negative]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

# Save both versions inside A3
output = Path("datasets")
output.mkdir(exist_ok=True)

v1.to_csv(output / "imdb_v1.csv", index=False)
v2.to_csv(output / "imdb_v2.csv", index=False)

print("Dataset V1:", len(v1), "reviews")
print("Dataset V2:", len(v2), "reviews")
print("Both dataset versions saved successfully!")
