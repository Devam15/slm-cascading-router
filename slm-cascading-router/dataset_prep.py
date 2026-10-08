"""
Baseline dataset preparation: MedHallu -> stratified pilot subset.

Run this on YOUR OWN machine or NUS HPC (needs real internet access to
huggingface.co — the sandbox this was written in couldn't reach it).

Install once:  pip install datasets pandas

What this does:
1. Downloads the MedHallu 'pqa_labeled' set (1,000 medical Q&A rows,
   each with a Ground Truth answer AND a synthetic Hallucinated Answer).
2. Prints how many rows fall into each Difficulty Level / Category of
   Hallucination, so you can see the dataset's actual shape before
   trusting any code that samples from it.
3. Builds a small, balanced pilot subset (default 120 rows) for your
   Week 2-3 pipeline testing. Change SUBSET_SIZE to ~300-400 when
   you're ready for the real Week 4 baseline run.
4. Saves it as a plain CSV you can open in Excel/Sheets to sanity-check
   by hand before running any model against it.
"""

from datasets import load_dataset
import pandas as pd

SUBSET_SIZE = 120  # pilot size; raise to 300-400 for the full Week 4 run
RANDOM_SEED = 42

print("Downloading MedHallu (pqa_labeled)...")
ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]
df = ds.to_pandas()
print(f"Loaded {len(df)} rows.")
print("\nColumns actually present in this dataset:")
print(list(df.columns))

# NOTE: column names below are based on the dataset viewer at the time
# this script was written (Question, Knowledge, Ground Truth,
# Difficulty Level, Hallucinated Answer, Category of Hallucination).
# Check the printed column list above FIRST — if any name below
# doesn't match exactly, fix the strata_cols list before continuing.
strata_cols = ["Difficulty Level", "Category of Hallucination"]

missing = [c for c in strata_cols if c not in df.columns]
if missing:
    raise SystemExit(
        f"These expected columns are missing: {missing}\n"
        f"Open the printed column list above, find the real names, "
        f"and edit strata_cols in this script to match."
    )

print("\nDifficulty Level breakdown:")
print(df["Difficulty Level"].value_counts())
print("\nCategory of Hallucination breakdown:")
print(df["Category of Hallucination"].value_counts())

# Stratified sample: pull roughly proportional counts from every
# (Difficulty Level x Category of Hallucination) combination, so the
# pilot subset isn't accidentally all-easy or all-one-category.
frac = SUBSET_SIZE / len(df)
subset = (
    df.groupby(strata_cols, group_keys=False)
    .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=RANDOM_SEED))
)
subset = subset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
subset = subset.head(SUBSET_SIZE)

out_path = "data/medhallu_pilot_subset.csv"
subset.to_csv(out_path, index=False)
print(f"\nSaved {len(subset)} rows to {out_path}")
print("Open this CSV and read ~20 rows by hand before writing any eval code.")
