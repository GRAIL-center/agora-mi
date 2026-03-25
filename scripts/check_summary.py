import json

with open("results/interference/clamp_summary_layer24_train.json") as f:
    d = json.load(f)

# Find the specific row for cap, alpha=0.2, condition=safe
for row in d["summary_rows"]:
    if row["alpha"] == 0.2 and row["mode"] == "cap" and row["condition"] == "safe":
        print("Safe condition stats:", row)

# Find the specific row for cap, alpha=0.2, condition=random
for row in d["summary_rows"]:
    if row["alpha"] == 0.2 and row["mode"] == "cap" and row["condition"] == "random":
        print("Random condition stats:", row)

# Find the paired test for safe_minus_random at alpha=0.2
for row in d["paired_tests_primary_metric"]:
    if row["alpha"] == 0.2 and row["mode"] == "cap" and row["compare"] == "safe_minus_random":
        print("Paired test stats (safe_minus_random):", row)
