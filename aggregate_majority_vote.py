#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import random
import math
from typing import List, Tuple


def bit_accuracy(pred: str, gt: str) -> Tuple[int, int, float]:
    """Compute the bitwise accuracy between two equal-length bitstrings."""
    if len(pred) != len(gt):
        raise ValueError(f"Length mismatch: pred={len(pred)}, gt={len(gt)}")
    correct = sum(1 for a, b in zip(pred, gt) if a == b)
    total = len(gt)
    return correct, total, correct / total


def majority_vote(bitstrings: List[str]) -> str:
    """Aggregate a list of binary strings using per-position majority vote."""
    if not bitstrings:
        raise ValueError("Input is empty; majority vote cannot be computed.")

    L = len(bitstrings[0])
    for s in bitstrings:
        if len(s) != L:
            raise ValueError("Found predicted_watermark values with inconsistent lengths.")
        if any(ch not in "01" for ch in s):
            raise ValueError(f"Found a non-binary string: {s}")

    voted = []
    n = len(bitstrings)

    for i in range(L):
        ones = sum(1 for s in bitstrings if s[i] == "1")
        zeros = n - ones
        # Break ties in favor of 1 by default.
        voted.append("1" if ones >= zeros else "0")

    return "".join(voted)


def load_rows(csv_path: str):
    """Load prediction rows from the input CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "predicted_watermark"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain at least {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            rows.append({
                "filename": row["filename"].strip(),
                "predicted_watermark": row["predicted_watermark"].strip(),
                "gt": (row.get("gt", "") or "").strip()
            })
    return rows


def find_csv_in_folder(input_path: str) -> str:
    """Resolve the CSV file path from either a file or a directory."""
    if os.path.isfile(input_path):
        return input_path

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    csv_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(".csv")
    ]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found under directory: {input_path}")
    if len(csv_files) > 1:
        print("[WARN] Multiple CSV files found; using the first one:", csv_files[0])

    return csv_files[0]


def main():
    """Run repeated random-sampling aggregation trials and save per-trial results."""
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample predicted watermarks, aggregate them with majority vote, "
            "and summarize the aggregated accuracy across repeated trials."
        )
    )
    parser.add_argument("--input", default="", help="Path to a CSV file, or a directory containing one.")
    parser.add_argument("--sample_size", type=int, default=30, help="Number of rows sampled per trial.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of repeated sampling trials.")
    parser.add_argument("--gt", type=str, default=None, help="Ground-truth watermark bitstring. If omitted, the script reads the first non-empty gt value from the CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_csv", type=str, default="", help="Path to save the per-trial aggregation results.")

    args = parser.parse_args()
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load predictions and resolve the ground-truth bitstring
    # ------------------------------------------------------------------
    csv_path = find_csv_in_folder(args.input)
    rows = load_rows(csv_path)

    if len(rows) < args.sample_size:
        raise ValueError(
            f"Not enough rows to sample {args.sample_size} items per trial; only {len(rows)} rows are available."
        )

    # Resolve the ground-truth bitstring.
    gt = args.gt
    if gt is None:
        gts = [r["gt"] for r in rows if r["gt"]]
        if not gts:
            raise ValueError("No --gt value was provided, and the CSV does not contain a usable gt column.")
        gt = gts[0]
        if any(x != gt for x in gts):
            print("[WARN] Multiple distinct gt values were found; using the first one.")

    # Validate bitstring lengths before sampling.
    L = len(rows[0]["predicted_watermark"])
    for r in rows:
        if len(r["predicted_watermark"]) != L:
            raise ValueError(f"{r['filename']} has a predicted_watermark with an inconsistent length.")
    if len(gt) != L:
        raise ValueError(f"gt has length {len(gt)}, but predicted_watermark has length {L}.")

    # ------------------------------------------------------------------
    # Run repeated aggregation trials
    # ------------------------------------------------------------------
    results = []
    agg_acc_list = []

    for trial_id in range(1, args.num_trials + 1):
        sampled = random.sample(rows, args.sample_size)
        sampled_preds = [r["predicted_watermark"] for r in sampled]
        sampled_files = [r["filename"] for r in sampled]

        voted = majority_vote(sampled_preds)
        correct_bits, total_bits, agg_acc = bit_accuracy(voted, gt)
        agg_acc_list.append(agg_acc)

        result = {
            "trial_id": trial_id,
            "sample_size": args.sample_size,
            "voted_watermark": voted,
            "correct_bits": correct_bits,
            "total_bits": total_bits,
            "agg_accuracy": agg_acc,
            "gt": gt,
            "sampled_filenames": ";".join(sampled_files),
        }
        results.append(result)

        # print(
        #     f"[Trial {trial_id:02d}] "
        #     f"agg_accuracy = {agg_acc:.6f} "
        #     f"({correct_bits}/{total_bits}) "
        #     f"voted = {voted}"
        # )

    mean_agg_acc = sum(agg_acc_list) / len(agg_acc_list)
    if len(agg_acc_list) > 1:
        variance = sum((x - mean_agg_acc) ** 2 for x in agg_acc_list) / len(agg_acc_list)
        std_agg_acc = math.sqrt(variance)
    else:
        variance = 0.0
        std_agg_acc = 0.0

    # ------------------------------------------------------------------
    # Summarize and save outputs
    # ------------------------------------------------------------------
    print("\n==================== Summary ====================")
    print(f"CSV file: {csv_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Sample size per trial: {args.sample_size}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Mean aggregated accuracy: {mean_agg_acc:.6f}")
    print(f"Aggregated accuracy variance: {variance:.6f}")
    print(f"Aggregated accuracy std: {std_agg_acc:.6f}")

    fieldnames = [
        "trial_id",
        "sample_size",
        "voted_watermark",
        "correct_bits",
        "total_bits",
        "agg_accuracy",
        "gt",
        "sampled_filenames"
    ]
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Per-trial results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
