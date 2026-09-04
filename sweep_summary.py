import pandas as pd

INPUT_FILE = "d3po_sweeps.csv"
OUTPUT_FILE = "d3po_sweeps_summary.csv"

df = pd.read_csv(INPUT_FILE)

group_cols = [
    "diversity_scale",
    "entropy_loss_coefficient",
    "learning_rate",
]

summary = (
    df.groupby(group_cols)["eval/hypervolume"]
    .agg(
        mean_hv="mean",
        std_hv="std",
        n_seeds="count",
        min_hv="min",
        max_hv="max",
    )
    .reset_index()
)

summary = summary.sort_values(
    by=["mean_hv", "std_hv"],
    ascending=[False, True],
)

summary["mean_plus_minus_std"] = summary.apply(
    lambda row: f"{row['mean_hv']:.2f} ± {row['std_hv']:.2f}",
    axis=1,
)

print(summary.to_string(index=False))

summary.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved summary to: {OUTPUT_FILE}")

print("\nBest configuration:")
print(summary.iloc[0].to_string())