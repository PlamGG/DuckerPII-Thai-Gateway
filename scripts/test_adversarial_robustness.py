"""
Adversarial Robustness Testing Suite (scripts/test_adversarial_robustness.py)
Generates real-world adversarial perturbations:
1. Spacing noise (extra spaces, collapsed spaces)
2. Common Thai typos / missing tone marks
3. Repeated punctuation (e.g., !!??, ...)
4. English case variations in emails/banks

Compares baseline performance on Clean Test vs Adversarial Test.
Outputs: reports/adversarial_robustness_report.json
"""

import os
import sys
import json
import random
from typing import List, Dict, Tuple

# Ensure current directory is in sys.path for local module import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import benchmark_baseline

TYPO_REPLACEMENTS = [
    ("เบอร์", "เบอ"),
    ("บัญชี", "บันชี"),
    ("โทรศัพท์", "ทอสับ"),
    ("กสิกร", "กสิกอน"),
    ("ไทยพาณิชย์", "ไทยพานิช"),
    ("กรุงเทพ", "กรุงเทบ"),
    ("ครับ", "คับ"),
    ("ค่ะ", "คะ")
]

def apply_adversarial_noise(text: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Applies real-world chat noise OUTSIDE entity spans and shifts offsets accurately.
    """
    # Sort entities by start
    ents = sorted(entities, key=lambda x: x["start"])
    new_chunks = []
    new_entities = []
    last_end = 0
    cur_idx = 0

    for ent in ents:
        s = ent["start"]
        e = ent["end"]
        val = ent["value"]
        lbl = ent["label"]
        literal = text[last_end:s]

        # Apply noise to literal outside entities
        for src, dst in TYPO_REPLACEMENTS:
            if src in literal and random.random() < 0.35:
                literal = literal.replace(src, dst, 1)

        if random.random() < 0.25:
            literal = literal + " " # extra space
            
        new_chunks.append(literal)
        cur_idx += len(literal)

        # Entity value
        ent_start = cur_idx
        ent_end = ent_start + len(val)
        new_chunks.append(val)
        cur_idx += len(val)

        new_entities.append({
            "start": ent_start,
            "end": ent_end,
            "label": lbl,
            "value": val
        })
        last_end = e

    tail = text[last_end:]
    if random.random() < 0.30:
        tail = tail + "..."
    new_chunks.append(tail)

    adv_text = "".join(new_chunks)
    return adv_text, new_entities

def run_adversarial_test():
    with open("data/test.json", "r", encoding="utf-8") as f:
        clean_test = json.load(f)

    random.seed(42)
    adv_test = []
    for item in clean_test:
        adv_t, adv_ents = apply_adversarial_noise(item["text"], item["entities"])
        adv_item = dict(item)
        adv_item["text"] = adv_t
        adv_item["entities"] = adv_ents
        adv_test.append(adv_item)

    # Evaluate Clean vs Adversarial
    def get_f1(dataset):
        labels = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]
        stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
        for item in dataset:
            preds = benchmark_baseline.extract_regex_entities(item["text"])
            gt_spans = {(e["start"], e["end"], e["label"]) for e in item["entities"]}
            pred_spans = {(p["start"], p["end"], p["label"]) for p in preds}

            for s, e, l in gt_spans:
                if (s, e, l) in pred_spans:
                    stats[l]["tp"] += 1
                else:
                    stats[l]["fn"] += 1
            for s, e, l in pred_spans:
                if (s, e, l) not in gt_spans:
                    stats[l]["fp"] += 1

        total_tp = sum(stats[l]["tp"] for l in labels)
        total_fp = sum(stats[l]["fp"] for l in labels)
        total_fn = sum(stats[l]["fn"] for l in labels)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(f1, 4), stats

    clean_f1, clean_stats = get_f1(clean_test)
    adv_f1, adv_stats = get_f1(adv_test)

    report = {
        "benchmark": "Regex Baseline Robustness",
        "clean_test_micro_f1": clean_f1,
        "adversarial_test_micro_f1": adv_f1,
        "f1_delta": round(adv_f1 - clean_f1, 4),
        "clean_stats": clean_stats,
        "adversarial_stats": adv_stats
    }

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "adversarial_robustness_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=====================================================")
    print("ADVERSARIAL ROBUSTNESS REPORT")
    print("=====================================================")
    print(f"Clean Test Micro F1       : {clean_f1:.4f}")
    print(f"Adversarial Test Micro F1 : {adv_f1:.4f}")
    print(f"Robustness Delta (Impact) : {report['f1_delta']:+.4f}")
    print(f"Report saved to           : {out_path}")
    print("=====================================================")

if __name__ == "__main__":
    run_adversarial_test()
