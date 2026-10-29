"""
Baseline Benchmark Evaluator: Regex Rule-Based Baseline
Tests the synthetic dataset against a realistic rule-based baseline (Regex)
and calculates Span-Level Precision, Recall, and F1-Score for each entity type.
Includes Non-Overlapping Disambiguation between PHONE and BANK_ACC.
"""

import json
import re
import sys
from typing import List, Dict, Tuple

# Regex Patterns for Baseline (Evaluated in Priority Order)
REGEX_PATTERNS = [
    # 1. Thai ID: 13 digits continuous or formatted with dashes
    ("ID_CARD", re.compile(r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b|\b\d{13}\b")),
    
    # 2. Email
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    
    # 3. Phone numbers: Thai formats (08x, 09x, 06x, +66)
    ("PHONE", re.compile(r"(\+66[-\s]?\d{1,2}[-\s]?\d{3}[-\s]?\d{4}|0[689]\d{1}[-\s]?\d{3}[-\s]?\d{4}|0[689]\d{8})")),
    
    # 4. Bank Account: 10-12 digits with or without dashes (exclude mobile prefixes 06, 08, 09)
    ("BANK_ACC", re.compile(r"\b\d{3}-\d{1}-\d{5}-\d{1}\b|\b(?!(?:0[689]))\d{10,12}\b")),
    
    # 5. Bank names (fixed list)
    ("BANK", re.compile(r"(กสิกรไทย|ไทยพาณิชย์|กรุงเทพ|กรุงไทย|กรุงศรีอยุธยา|ทหารไทยธนชาต|ออมสิน|กสิกร|SCB|KBANK|KTB|TTB|CIMB)")),
    
    # 6. Name pattern: heuristic based on common Thai titles
    ("NAME", re.compile(r"(นาย|นาง|นางสาว|คุณ|น\.ส\.)[ก-๙]+\s+[ก-๙]+"))
]

def extract_regex_entities(text: str) -> List[Dict]:
    preds = []
    occupied_spans = []  # To avoid multi-label overlap on the same characters

    for label, pattern in REGEX_PATTERNS:
        for match in pattern.finditer(text):
            s, e = match.span()
            # Check overlap with already extracted spans
            if any(max(s, occ_s) < min(e, occ_e) for occ_s, occ_e in occupied_spans):
                continue
                
            preds.append({
                "start": s,
                "end": e,
                "label": label,
                "value": match.group()
            })
            occupied_spans.append((s, e))
            
    # Sort by start offset
    preds.sort(key=lambda x: x["start"])
    return preds

def evaluate_spans(ground_truth_list: List[Dict], prediction_list: List[Dict]):
    """
    Span-Level Exact Match Evaluation (Precision, Recall, F1)
    """
    labels = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]
    stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
    
    for gt_item, pred_item in zip(ground_truth_list, prediction_list):
        gt_spans = {(e["start"], e["end"], e["label"]) for e in gt_item["entities"]}
        pred_spans = {(e["start"], e["end"], e["label"]) for e in pred_item}
        
        # True Positives
        tp_set = gt_spans.intersection(pred_spans)
        for s, e, l in tp_set:
            if l in stats:
                stats[l]["tp"] += 1
                
        # False Positives (predicted but not in GT)
        fp_set = pred_spans - gt_spans
        for s, e, l in fp_set:
            if l in stats:
                stats[l]["fp"] += 1
                
        # False Negatives (in GT but missed by prediction)
        fn_set = gt_spans - pred_spans
        for s, e, l in fn_set:
            if l in stats:
                stats[l]["fn"] += 1

    # Print Report
    print(f"{'Entity':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support (GT)':<12}")
    print("-" * 65)
    
    total_tp = sum(stats[l]["tp"] for l in labels)
    total_fp = sum(stats[l]["fp"] for l in labels)
    total_fn = sum(stats[l]["fn"] for l in labels)
    
    f1_list = []
    
    for l in labels:
        tp = stats[l]["tp"]
        fp = stats[l]["fp"]
        fn = stats[l]["fn"]
        support = tp + fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if support > 0:
            f1_list.append(f1)
            
        print(f"{l:<12} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {support:<12}")
        
    print("-" * 65)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    print(f"{'Macro Avg':<12} | {'-':<10} | {'-':<10} | {macro_f1:<10.4f} | {sum(stats[l]['tp'] + stats[l]['fn'] for l in labels):<12}")
    print(f"{'Micro Avg':<12} | {micro_prec:<10.4f} | {micro_rec:<10.4f} | {micro_f1:<10.4f} | {sum(stats[l]['tp'] + stats[l]['fn'] for l in labels):<12}")

if __name__ == "__main__":
    split_file = sys.argv[1] if len(sys.argv) > 1 else "data/test.json"
    with open(split_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    predictions = [extract_regex_entities(item["text"]) for item in ground_truth]
    evaluate_spans(ground_truth, predictions)
