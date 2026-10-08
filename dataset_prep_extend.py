"""
Extend the 120-row pilot subset to a larger stratified sample, GUARANTEEING
every row from the original 120-row pilot is included in the new, bigger
set (not just probabilistically likely to be, via a lucky random seed).

HONEST NOTE: this does NOT save you any model-computation time. None of
drift_probe.py / uncertain_expression_probe.py / router_demo.py cache or
reuse old outputs -- running any of them on the new bigger file re-queries
the model for every row, including the original 120. Keeping the original
120 as a literal subset here is about experimental-design consistency (the
same "core" 120 questions stay part of every larger set from here on), not
a compute shortcut.

Run this on YOUR OWN machine (needs real internet access to huggingface.co).
Install once: pip install datasets pandas
"""

import pandas as pd

TARGET_SIZE = 350                              # new total size -- adjust as needed
EXISTING_CSV = "data/medhallu_pilot_subset.csv"     # your original 120-row file
OUTPUT_CSV = f"data/medhallu_pilot_subset_{TARGET_SIZE}.csv"
RANDOM_SEED = 42
STRATA_COLS = ["Difficulty Level", "Category of Hallucination"]
MATCH_COL = "Question"   # assumed unique per row -- used to identify which
                         # full-dataset rows correspond to your existing 120


def build_extended_subset(full_df: pd.DataFrame, existing_df: pd.DataFrame,
                           target_size: int, strata_cols: list, match_col: str,
                           seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Core logic, separated out so it can be unit-tested without needing
    the real MedHallu download or the `datasets` library.
    """
    matched_mask = full_df[match_col].isin(existing_df[match_col])
    n_matched = int(matched_mask.sum())
    if n_matched != len(existing_df):
        print(f"[!] WARNING: only {n_matched} of {len(existing_df)} existing rows "
              f"were found in the full dataset by matching on '{match_col}'. "
              f"Continuing, but double check EXISTING_CSV/MATCH_COL are correct "
              f"before trusting the result -- this usually means a different "
              f"dataset version, or duplicate/altered Question text.")

    remaining_pool = full_df[~matched_mask].copy()
    n_needed = target_size - len(existing_df)
    if n_needed <= 0:
        raise ValueError(
            f"target_size ({target_size}) is not bigger than the existing "
            f"subset ({len(existing_df)} rows) -- nothing to add."
        )
    if n_needed > len(remaining_pool):
        raise ValueError(
            f"Need {n_needed} more rows but only {len(remaining_pool)} remain "
            f"in the pool after excluding existing rows."
        )

    missing_strata = [c for c in strata_cols if c not in remaining_pool.columns]
    if missing_strata:
        raise ValueError(f"Missing strata columns in remaining pool: {missing_strata}")

    frac = n_needed / len(remaining_pool)
    new_rows = (
        remaining_pool.groupby(strata_cols, group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
    )
    new_rows = new_rows.sample(frac=1, random_state=seed).reset_index(drop=True)
    new_rows = new_rows.head(n_needed)

    if len(new_rows) < n_needed:
        # Rounding in the per-group sampling above can occasionally undershoot
        # by a row or two -- top up randomly from whatever's left in the pool.
        shortfall = n_needed - len(new_rows)
        already_picked = full_df[match_col].isin(new_rows[match_col])
        topup_pool = remaining_pool[~remaining_pool[match_col].isin(new_rows[match_col])]
        topup = topup_pool.sample(shortfall, random_state=seed)
        new_rows = pd.concat([new_rows, topup], ignore_index=True)

    combined = pd.concat([existing_df, new_rows], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return combined


def main():
    from datasets import load_dataset

    print("Downloading full MedHallu (pqa_labeled)...")
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")["train"]
    full_df = ds.to_pandas()
    print(f"Loaded {len(full_df)} total rows.")

    print(f"\nLoading existing pilot subset from {EXISTING_CSV}...")
    existing_df = pd.read_csv(EXISTING_CSV)
    print(f"Existing subset has {len(existing_df)} rows.")

    combined = build_extended_subset(
        full_df, existing_df, TARGET_SIZE, STRATA_COLS, MATCH_COL, RANDOM_SEED
    )

    combined.to_csv(OUTPUT_CSV, index=False)
    n_new = len(combined) - len(existing_df)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_CSV} "
          f"({len(existing_df)} original + {n_new} new).")
    print("Open this CSV and spot-check that your original 120 questions are "
          "still present (search for a couple you recognize), and that the "
          "new rows look reasonable, before running any model against it.")


if __name__ == "__main__":
    main()
