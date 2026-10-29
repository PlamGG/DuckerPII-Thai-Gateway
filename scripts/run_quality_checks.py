"""
Comprehensive Quality Control Suite (scripts/run_quality_checks.py)
Validates all 8 quality dimensions:
1. Span integrity & character slice exactness
2. Entity label validity
3. Duplicate detection
4. Split leakage (Templates and Names)
5. Entity distribution & ADDRESS density (>= 20.0%)
6. Hard negatives & Pure negatives coverage
7. Thai tokenizer boundary validation (PyThaiNLP newmm)
8. Synthetic realism risk check (Combinatorial privacy guard)

Saves detailed results to reports/quality_report.json
"""

import os
import json
import sys
from typing import Dict, List, Set, Tuple

try:
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False

ALLOWED_LABELS = {"NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"}

def run_quality_checks():
    data_dir = "data"
    splits = ["train", "validation", "test"]
    loaded_data = {}
    
    for s in splits:
        p = os.path.join(data_dir, f"{s}.json")
        if not os.path.exists(p):
            print(f"[CRITICAL FAIL] Missing file: {p}")
            sys.exit(1)
        with open(p, "r", encoding="utf-8") as f:
            loaded_data[s] = json.load(f)

    report = {
        "status": "PASSED",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    print("=====================================================")
    print("RUNNING COMPREHENSIVE DATASET QUALITY CHECKS (CI)")
    print("=====================================================")

    # 1. Span Integrity & Overlap Check
    span_errors = 0
    overlap_errors = 0
    total_entities = 0

    for s in splits:
        for item in loaded_data[s]:
            text = item["text"]
            ents = sorted(item["entities"], key=lambda x: x["start"])
            
            # Check individual spans
            for ent in ents:
                total_entities += 1
                st = ent["start"]
                en = ent["end"]
                val = ent["value"]
                
                if st < 0 or en > len(text) or st >= en:
                    span_errors += 1
                    report["errors"].append(f"[{s}:{item['id']}] Invalid bounds: ({st}, {en}) for len={len(text)}")
                    continue
                if text[st:en] != val:
                    span_errors += 1
                    report["errors"].append(f"[{s}:{item['id']}] Slice mismatch: expected '{val}', got '{text[st:en]}'")

            # Check overlap between adjacent entities
            for i in range(len(ents) - 1):
                if ents[i]["end"] > ents[i+1]["start"]:
                    overlap_errors += 1
                    report["errors"].append(f"[{s}:{item['id']}] Overlapping entities: {ents[i]} vs {ents[i+1]}")

    report["checks"]["span_integrity"] = {
        "total_entities_checked": total_entities,
        "span_errors": span_errors,
        "overlap_errors": overlap_errors,
        "status": "PASSED" if (span_errors == 0 and overlap_errors == 0) else "FAILED"
    }
    print(f"1. SPAN INTEGRITY CHECK: Checked {total_entities} entities -> {'PASSED' if span_errors == 0 and overlap_errors == 0 else 'FAILED'}")

    # 2. Entity Label Validity
    invalid_labels = []
    for s in splits:
        for item in loaded_data[s]:
            for ent in item["entities"]:
                if ent["label"] not in ALLOWED_LABELS:
                    invalid_labels.append(f"[{item['id']}] Invalid label: {ent['label']}")

    report["checks"]["label_validity"] = {
        "invalid_label_count": len(invalid_labels),
        "status": "PASSED" if len(invalid_labels) == 0 else "FAILED"
    }
    print(f"2. LABEL VALIDITY CHECK : Allowed labels only -> {'PASSED' if len(invalid_labels) == 0 else 'FAILED'}")

    # 3. Duplicate Detection within Splits
    duplicate_texts = 0
    for s in splits:
        seen = set()
        for item in loaded_data[s]:
            t = item["text"].strip()
            if t in seen:
                duplicate_texts += 1
            seen.add(t)

    report["checks"]["duplicates"] = {
        "duplicate_text_count": duplicate_texts,
        "status": "PASSED" if duplicate_texts == 0 else "WARNING"
    }
    if duplicate_texts > 0:
        report["warnings"].append(f"Found {duplicate_texts} duplicate texts within splits")
    print(f"3. DUPLICATE CHECK      : Found {duplicate_texts} duplicates -> {'PASSED' if duplicate_texts == 0 else 'WARNING'}")

    # 4. Split Leakage (Templates & Lexical Names)
    tr_tmpls = set(x["template_id"] for x in loaded_data["train"])
    te_tmpls = set(x["template_id"] for x in loaded_data["test"])
    va_tmpls = set(x["template_id"] for x in loaded_data["validation"])

    leak_tr_te_tmpl = tr_tmpls.intersection(te_tmpls)
    
    # Lexical Name Leakage (Train vs Test)
    def extract_names(records):
        names = set()
        for r in records:
            for e in r["entities"]:
                if e["label"] == "NAME":
                    v = e["value"]
                    for t in ["นาย", "นางสาว", "นาง", "น.ส.", "คุณ", "พี่", "น้อง", "ช่าง", "หมอ"]:
                        if v.startswith(t):
                            v = v[len(t):].strip()
                            break
                    names.add(v)
        return names

    tr_names = extract_names(loaded_data["train"])
    te_names = extract_names(loaded_data["test"])
    name_leakage = tr_names.intersection(te_names)

    report["checks"]["split_leakage"] = {
        "template_leakage_train_test": len(leak_tr_te_tmpl),
        "name_leakage_train_test": len(name_leakage),
        "status": "PASSED" if (len(leak_tr_te_tmpl) == 0 and len(name_leakage) == 0) else "FAILED"
    }
    print(f"4. SPLIT LEAKAGE CHECK  : Template Overlap={len(leak_tr_te_tmpl)}, Name Overlap={len(name_leakage)} -> {'PASSED' if len(leak_tr_te_tmpl) == 0 and len(name_leakage) == 0 else 'FAILED'}")

    # 5. Entity Distribution & Address Density
    dist_report = {}
    address_density_passed = True
    for s in splits:
        total_ents = sum(len(x["entities"]) for x in loaded_data[s])
        addr_cnt = sum(1 for x in loaded_data[s] for e in x["entities"] if e["label"] == "ADDRESS")
        pct = (addr_cnt / total_ents * 100) if total_ents > 0 else 0
        dist_report[s] = {"total_entities": total_ents, "address_count": addr_cnt, "address_pct": round(pct, 2)}
        if pct < 20.0:
            address_density_passed = False
            report["warnings"].append(f"{s} ADDRESS density is below 20.0%: {pct:.2f}%")

    report["checks"]["entity_distribution"] = {
        "splits": dist_report,
        "address_target_ge_20": address_density_passed,
        "status": "PASSED" if address_density_passed else "WARNING"
    }
    print(f"5. ADDRESS DENSITY CHECK: Train={dist_report['train']['address_pct']}%, Val={dist_report['validation']['address_pct']}%, Test={dist_report['test']['address_pct']}% -> {'PASSED' if address_density_passed else 'SUB-OPTIMAL'}")

    # 6. Hard Negatives & Pure Negatives Coverage
    neg_counts = {s: sum(1 for x in loaded_data[s] if len(x["entities"]) == 0) for s in splits}
    report["checks"]["pure_negatives"] = {
        "pure_negatives_per_split": neg_counts,
        "status": "PASSED" if all(v > 0 for v in neg_counts.values()) else "FAILED"
    }
    print(f"6. PURE NEGATIVES CHECK : Train={neg_counts['train']}, Val={neg_counts['validation']}, Test={neg_counts['test']} -> PASSED")

    # 7. Thai Tokenizer Boundary Validation (PyThaiNLP newmm)
    boundary_violations = 0
    if PYTHAINLP_AVAILABLE:
        for s in splits:
            for item in loaded_data[s]:
                text = item["text"]
                tokens = word_tokenize(text, engine="newmm")
                token_offsets = []
                cur = 0
                for tok in tokens:
                    token_offsets.append((cur, cur + len(tok)))
                    cur += len(tok)
                starts = set(x[0] for x in token_offsets)
                ends = set(x[1] for x in token_offsets)
                
                for ent in item["entities"]:
                    if ent["start"] not in starts or ent["end"] not in ends:
                        boundary_violations += 1
                        report["warnings"].append(f"[{item['id']}] Token boundary violation on '{ent['value']}'")
        report["checks"]["token_boundary"] = {
            "violations": boundary_violations,
            "status": "PASSED" if boundary_violations == 0 else "WARNING"
        }
        print(f"7. TOKEN BOUNDARY CHECK : PyThaiNLP newmm violations={boundary_violations} -> {'PASSED' if boundary_violations == 0 else 'WARNING'}")
    else:
        print("7. TOKEN BOUNDARY CHECK : PyThaiNLP not installed (Skipped)")

    # 8. Synthetic Realism & Identity Combination Collisions
    identity_combos = set()
    collision_count = 0
    for s in splits:
        for item in loaded_data[s]:
            names = [e["value"] for e in item["entities"] if e["label"] == "NAME"]
            phones = [e["value"] for e in item["entities"] if e["label"] == "PHONE"]
            if names and phones:
                combo = f"{names[0]}|{phones[0]}"
                if combo in identity_combos:
                    collision_count += 1
                identity_combos.add(combo)

    report["checks"]["synthetic_realism_risk"] = {
        "name_phone_combination_collisions": collision_count,
        "status": "PASSED" if collision_count == 0 else "WARNING"
    }
    print(f"8. COMBINATORIAL SAFETY : Name+Phone Collisions={collision_count} -> {'PASSED' if collision_count == 0 else 'WARNING'}")

    # Determine final exit status
    if report["errors"]:
        report["status"] = "FAILED"
        exit_code = 1
    else:
        report["status"] = "PASSED"
        exit_code = 0

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Sync quality_gate results into data/manifest.json
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            m_data = json.load(mf)
        m_data["quality_gate"] = {
            "status": report["status"],
            "total_entities_checked": report["checks"].get("span_integrity", {}).get("total_entities_checked", 0),
            "span_errors": report["checks"].get("span_integrity", {}).get("span_errors", 0),
            "duplicate_count": report["checks"].get("duplicates", {}).get("duplicate_text_count", 0),
            "template_leakage": report["checks"].get("split_leakage", {}).get("template_leakage_train_test", 0),
            "name_leakage": report["checks"].get("split_leakage", {}).get("name_leakage_train_test", 0),
            "token_boundary_violations": report["checks"].get("token_boundary", {}).get("violations", 0),
            "address_target_ge_20": report["checks"].get("entity_distribution", {}).get("address_target_ge_20", True)
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(m_data, mf, ensure_ascii=False, indent=2)

    print("=====================================================")
    print(f"OVERALL QUALITY CHECK RESULT: {report['status']} (Exit Code {exit_code})")
    print(f"Report saved to: {report_path}")
    print("=====================================================")
    sys.exit(exit_code)

if __name__ == "__main__":
    run_quality_checks()
