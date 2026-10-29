"""
Comprehensive Model & Baseline Evaluation Suite (scripts/evaluate_model.py)
Evaluates on Exact Match, Partial Overlap, and Character-Level metrics.
Outputs detailed analysis to reports/evaluation_report.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Set, Tuple

# Ensure current directory is in sys.path for local module import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import benchmark_baseline

def compute_char_f1(gt_text: str, gt_spans: List[Dict], pred_spans: List[Dict]) -> Tuple[float, float, float]:
    """
    Computes character-level precision, recall, and F1 over entity tokens.
    """
    gt_chars = set()
    for ent in gt_spans:
        for idx in range(ent["start"], ent["end"]):
            gt_chars.add((idx, ent["label"]))

    pred_chars = set()
    for ent in pred_spans:
        for idx in range(ent["start"], ent["end"]):
            pred_chars.add((idx, ent["label"]))

    tp = len(gt_chars.intersection(pred_chars))
    fp = len(pred_chars - gt_chars)
    fn = len(gt_chars - pred_chars)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def evaluate_comprehensive(dataset_path: str = "data/test.json", output_report_path: str = "reports/evaluation_report.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)

    labels = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]
    
    # Exact Match Stats
    exact_stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
    # Partial Overlap Stats (Intersection > 0 and same label)
    partial_stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}

    # Error analysis containers
    missed_entities = []
    false_positives = []
    boundary_errors = []
    label_confusions = []

    total_char_tp = 0
    total_char_fp = 0
    total_char_fn = 0

    for sample in ground_truth_list:
        text = sample["text"]
        gt_ents = sample["entities"]
        pred_ents = benchmark_baseline.extract_regex_entities(text)

        # Character Level
        p_c, r_c, f_c = compute_char_f1(text, gt_ents, pred_ents)

        # Exact Sets
        gt_exact = {(e["start"], e["end"], e["label"], e["value"]) for e in gt_ents}
        pred_exact = {(p["start"], p["end"], p["label"], p["value"]) for p in pred_ents}

        # Exact Match Calculation
        for ge in gt_exact:
            s, e, l, v = ge
            if ge in pred_exact:
                exact_stats[l]["tp"] += 1
            else:
                exact_stats[l]["fn"] += 1

        for pe in pred_exact:
            s, e, l, v = pe
            if pe not in gt_exact:
                exact_stats[l]["fp"] += 1

        # Partial Overlap & Boundary Analysis
        matched_gt_indices = set()
        matched_pred_indices = set()

        for g_idx, g in enumerate(gt_ents):
            found_partial = False
            for p_idx, p in enumerate(pred_ents):
                # Check overlap: max(s1, s2) < min(e1, e2)
                overlap = max(g["start"], p["start"]) < min(g["end"], p["end"])
                if overlap:
                    if g["label"] == p["label"]:
                        found_partial = True
                        matched_gt_indices.add(g_idx)
                        matched_pred_indices.add(p_idx)
                        # Check boundary error (same label, but start or end mismatch)
                        if (g["start"], g["end"]) != (p["start"], p["end"]):
                            boundary_errors.append({
                                "text": text,
                                "label": g["label"],
                                "gt_span": (g["start"], g["end"], g["value"]),
                                "pred_span": (p["start"], p["end"], p["value"])
                            })
                    else:
                        label_confusions.append({
                            "text": text,
                            "gt_label": g["label"],
                            "pred_label": p["label"],
                            "value": g["value"]
                        })
            if found_partial:
                partial_stats[g["label"]]["tp"] += 1
            else:
                partial_stats[g["label"]]["fn"] += 1
                missed_entities.append({"id": sample["id"], "text": text, "entity": g})

        for p_idx, p in enumerate(pred_ents):
            if p_idx not in matched_pred_indices:
                partial_stats[p["label"]]["fp"] += 1
                false_positives.append({"id": sample["id"], "text": text, "pred": p})

    # Compile Per-Entity Metrics
    per_entity_exact = {}
    per_entity_partial = {}

    exact_f1s = []
    partial_f1s = []

    for l in labels:
        tp = exact_stats[l]["tp"]
        fp = exact_stats[l]["fp"]
        fn = exact_stats[l]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if (tp + fn) > 0:
            exact_f1s.append(f1)
        per_entity_exact[l] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}

        # Partial
        ptp = partial_stats[l]["tp"]
        pfp = partial_stats[l]["fp"]
        pfn = partial_stats[l]["fn"]
        pprec = ptp / (ptp + pfp) if (ptp + pfp) > 0 else 0.0
        prec_r = ptp / (ptp + pfn) if (ptp + pfn) > 0 else 0.0
        pf1 = (2 * pprec * prec_r) / (pprec + prec_r) if (pprec + prec_r) > 0 else 0.0
        if (ptp + pfn) > 0:
            partial_f1s.append(pf1)
        per_entity_partial[l] = {"precision": round(pprec, 4), "recall": round(prec_r, 4), "f1": round(pf1, 4), "support": ptp + pfn}

    total_exact_tp = sum(exact_stats[l]["tp"] for l in labels)
    total_exact_fp = sum(exact_stats[l]["fp"] for l in labels)
    total_exact_fn = sum(exact_stats[l]["fn"] for l in labels)

    micro_prec = total_exact_tp / (total_exact_tp + total_exact_fp) if (total_exact_tp + total_exact_fp) > 0 else 0.0
    micro_rec = total_exact_tp / (total_exact_tp + total_exact_fn) if (total_exact_tp + total_exact_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(exact_f1s) / len(exact_f1s) if exact_f1s else 0.0

    eval_report = {
        "dataset": dataset_path,
        "model": "regex_baseline_v4",
        "metrics": {
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "exact_match_f1": round(micro_f1, 4),
            "partial_overlap_f1": round(sum(partial_f1s) / len(partial_f1s), 4) if partial_f1s else 0.0,
            "boundary_error_count": len(boundary_errors),
            "label_confusion_count": len(label_confusions)
        },
        "per_entity_exact_match": per_entity_exact,
        "per_entity_partial_overlap": per_entity_partial,
        "error_analysis": {
            "missed_entities_count": len(missed_entities),
            "false_positives_count": len(false_positives),
            "sample_boundary_errors": boundary_errors[:5],
            "sample_label_confusions": label_confusions[:5]
        }
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    # Print Summary Table
    print("==========================================================================")
    print("BASELINE EVALUATION REPORT (EXACT MATCH vs PARTIAL OVERLAP)")
    print("==========================================================================")
    print(f"{'Entity':<10} | {'Exact P':<8} | {'Exact R':<8} | {'Exact F1':<8} | {'Partial F1':<10} | {'Support':<8}")
    print("-" * 74)
    for l in labels:
        ex = per_entity_exact[l]
        pt = per_entity_partial[l]
        print(f"{l:<10} | {ex['precision']:<8.4f} | {ex['recall']:<8.4f} | {ex['f1']:<8.4f} | {pt['f1']:<10.4f} | {ex['support']:<8}")
    print("-" * 74)
    print(f"Overall Exact Micro F1  : {micro_f1:.4f}")
    print(f"Overall Exact Macro F1  : {macro_f1:.4f}")
    print(f"Partial Overlap Macro F1: {sum(partial_f1s) / len(partial_f1s):.4f}")
    print(f"Boundary Errors Detected: {len(boundary_errors)}")
    print(f"Full Report saved to    : {output_report_path}")
    print("==========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test.json", help="Path to evaluation dataset")
    parser.add_argument("--output", default="reports/evaluation_report.json", help="Output path for JSON report")
    args = parser.parse_args()
    evaluate_comprehensive(args.data, args.output)
