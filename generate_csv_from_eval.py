import os
import json
import csv

BASE_DIR = "."

OUTPUT_CSV = "eval_summary.csv"

FIELDS = [
    "global_best_conf",
    "map50",
    "map75",
    "map50_95",
    "f1_f1_50",
    "f1_f1_75",
    "precision_50",
    "precision_75",
    "recall_50",
    "recall_75",
]

rows = []

for folder in os.listdir(BASE_DIR):
    if folder.startswith("eval_"):
        json_path = os.path.join(BASE_DIR, folder, "results_on_test.json")

        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract metrics
            supervision = data.get("supervision_metrics", {})
            
            # Build row automatically from FIELDS list
            row = {"model_name": folder}

            for field in FIELDS:
                # Some fields are at root ("global_best_conf"),
                # others inside supervision_metrics
                if field in data:
                    row[field] = data.get(field)
                else:
                    row[field] = supervision.get(field)

            rows.append(row)


with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["model_name"] + FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV written to {OUTPUT_CSV} with {len(rows)} rows.")
