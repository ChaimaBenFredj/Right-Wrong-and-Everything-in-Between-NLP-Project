# This file Samples scenarios from the ETHICS benchmark dataset (Hendrycks et al., 2021) for moral alignment evaluation of LLMs.

import json
import os
import random
from datasets import load_dataset


# ------------------------- Config -----------------------------

random.seed(42)

SUBSETS= ["commonsense", "justice", "deontology"]
N_PER_SUBSET = 25  

LABEL_MAP = {
    "commonsense": {0: "acceptable",   1: "unacceptable"},  
    "justice": {0: "unacceptable", 1: "acceptable"},    
    "deontology": {0: "unacceptable", 1: "acceptable"},   
}


#---------------------- Text extraction -----------------------------------

def extract_text(example, subset) :
    """
    Extracts scenario text from ethics dataset.
    """
    if subset == "deontology":
        scenario = example.get("scenario", "").strip()
        excuse = example.get("excuse",   "").strip()
        if scenario and excuse:
            return f"{scenario} {excuse}"
        return scenario or excuse

    # commonsense and justice,one text field
    for field in ["input", "scenario", "text", "sentence"]:
        val = example.get(field, "")
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key, val in example.items():
        if key != "label" and isinstance(val, str) and val.strip():
            return val.strip()

    return ""


# -----------------------------Sampling ----------------------

def sample_subset(subset_name, n):
    """returns balanced examples."""
    print(f"  Loading ETHICS / {subset_name}  (test split)...")
    dataset = load_dataset("hendrycks/ethics", subset_name, split="test")

    # Show available fields for transparency
    if len(dataset) > 0:
        fields = list(dataset[0].keys())
        print(f"    Fields available : {fields}")

    lmap = LABEL_MAP[subset_name]
    positives = [x for x in dataset if lmap[x["label"]] == "acceptable"]
    negatives = [x for x in dataset if lmap[x["label"]] == "unacceptable"]

    print(f"    Class balance    : {len(positives)} acceptable / "
          f"{len(negatives)} unacceptable")

    n_pos= (n + 1) // 2
    n_neg = n // 2

    sampled = (
        random.sample(positives, min(n_pos, len(positives))) +
        random.sample(negatives, min(n_neg, len(negatives)))
    )
    random.shuffle(sampled)
    return sampled


def build_ethics_sample() :
    """Builds the full sample for all subsets"""
    records = []

    for subset in SUBSETS:
        sampled = sample_subset(subset, N_PER_SUBSET)
        lmap    = LABEL_MAP[subset]

        empty_count = 0
        for i, example in enumerate(sampled):
            text = extract_text(example, subset)
            lbl  = example["label"]

            if not text:
                empty_count += 1

            records.append({
                "id":                 f"ethics_{subset}_{i+1:02d}",
                "subset":             subset,
                "text":               text,
                "ground_truth":       lbl,
                "ground_truth_label": lmap[lbl],
            })

        subset_records = [r for r in records if r["subset"] == subset]
        n_acc   = sum(1 for r in subset_records if r["ground_truth_label"] == "acceptable")
        n_unacc = sum(1 for r in subset_records if r["ground_truth_label"] == "unacceptable")
        print(f"    Sampled          : {len(subset_records)} "
              f"({n_acc} acceptable / {n_unacc} unacceptable)")
        if empty_count:
            print(f"    WARNING          : {empty_count} records had empty text!")
        print()

    return records


# -------------------------- Main --------------------------------

def main():
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base, "data", "ethics_sample.json")

    print("=" * 55)
    print("  ETHICS Benchmark — Scenario Sampling")
    print(f"  Subsets      : {', '.join(SUBSETS)}")
    print(f"  Per subset   : {N_PER_SUBSET}  (balanced by label)")
    print(f"  Total        : {N_PER_SUBSET * len(SUBSETS)} scenarios")
    print(f"  API calls    : {N_PER_SUBSET * len(SUBSETS) * 3}  (x3 models)")
    print("=" * 55 + "\n")

    records = build_ethics_sample()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("=" * 55)
    print(f"Saved {len(records)} scenarios to data/ethics_sample.json\n")
    print(f"  {'Subset':<14} {'Total':>6} {'Acceptable':>12} {'Unacceptable':>14}")
    print("  " + "-" * 47)
    for subset in SUBSETS:
        items   = [r for r in records if r["subset"] == subset]
        n_acc   = sum(1 for r in items if r["ground_truth_label"] == "acceptable")
        n_unacc = sum(1 for r in items if r["ground_truth_label"] == "unacceptable")
        print(f"  {subset:<14} {len(items):>6} {n_acc:>12} {n_unacc:>14}")

    # Preview one example from each subset
    print(f"\nPreview (one example per subset):")
    for subset in SUBSETS:
        r = next(x for x in records if x["subset"] == subset)
        print(f"\n  [{r['id']}]  label: {r['ground_truth_label']}")
        print(f"  text : {r['text'][:120]}...")

if __name__ == "__main__":
    main()
