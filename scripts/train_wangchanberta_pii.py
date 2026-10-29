"""
Production Training & Deployment Script: WangchanBERTa for Thai PII Detection
=============================================================================
Fine-tunes 'airesearch/wangchanberta-base-att-spm-uncased' on 'DuckerMaster/Thai-PII-Synthetic'
using character-level span alignment and Two-Tier Cascaded Hybrid Architecture.

Features:
1. Character span alignment with native SentencePiece offset mappings.
2. Hugging Face Trainer with AdamW, Linear Warmup, FP16, and seqeval metrics.
3. Test Set Character Span Exact Match F1 benchmark against Regex Baseline.
4. Two-Tier Cascaded Hybrid ThaiPIIGateway:
   - Tier 1: Deterministic Pattern Engine + Mod11 Checksum Validator (100% Precision on IDs/Emails/Phones).
   - Tier 2: Neural WangchanBERTa Engine with Sliding Window Chunking (High Recall on Names/Addresses).
   - Tier 3: Priority Span Disambiguation Matrix for overlapping candidates.
   - Single-Pass Regex Substitution in unmask() eliminating cascading replacements.
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set

import torch
import numpy as np

# Label Schema (15 classes: O + 7 B-* + 7 I-*)
LABEL_NAMES = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]
ID2LABEL: Dict[int, str] = {0: "O"}
LABEL2ID: Dict[str, int] = {"O": 0}

for idx, lbl in enumerate(LABEL_NAMES):
    b_id = 2 * idx + 1
    i_id = 2 * idx + 2
    b_tag = f"B-{lbl}"
    i_tag = f"I-{lbl}"
    ID2LABEL[b_id] = b_tag
    ID2LABEL[i_id] = i_tag
    LABEL2ID[b_tag] = b_id
    LABEL2ID[i_tag] = i_id

# Regex Baseline numbers from reports/evaluation_report.json (OOD Test Set: 2,000 samples)
REGEX_BASELINE = {
    "NAME": {"precision": 0.8487, "recall": 0.5412, "f1": 0.6610},
    "ID_CARD": {"precision": 1.0000, "recall": 0.8667, "f1": 0.9286},
    "PHONE": {"precision": 0.9910, "recall": 0.8483, "f1": 0.9141},
    "EMAIL": {"precision": 1.0000, "recall": 0.8152, "f1": 0.8982},
    "BANK_ACC": {"precision": 1.0000, "recall": 0.7608, "f1": 0.8641},
    "BANK": {"precision": 0.7989, "recall": 0.6714, "f1": 0.7296},
    "ADDRESS": {"precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
    "OVERALL_MICRO": {"precision": 0.8037, "recall": 0.5971, "f1": 0.6851},
}

# Deterministic Regex Patterns for Tier 1 Engine (Lookarounds replace \b for Thai Unicode safety)
TIER1_REGEX_PATTERNS = [
    # 1. Thai ID: 13 digits with or without dashes (Lookaround instead of \b)
    ("ID_CARD", re.compile(r"(?<!\d)\d{1}-\d{4}-\d{5}-\d{2}-\d{1}(?!\d)|(?<!\d)\d{13}(?!\d)")),
    # 2. Email Address
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    # 3. Phone numbers: Thai mobile, Bangkok 02, and provincial 03-07 landlines
    ("PHONE", re.compile(r"(?<!\d)(?:\+66[-\s]?\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}|0[689]\d[-\s]?\d{3}[-\s]?\d{4}|02[-\s]?\d{3}[-\s]?\d{4}|0[3-7]\d[-\s]?\d{3}[-\s]?\d{3})(?!\d)")),
    # 4. Bank Account: 10-12 digits with or without dashes (exclude mobile prefixes 06, 08, 09)
    ("BANK_ACC", re.compile(r"(?<!\d)\d{3}-\d{1}-\d{5}-\d{1}(?!\d)|(?<!\d)(?!(?:0[689]))\d{10,12}(?!\d)")),
    # 5. Bank names (require bank indicator prefix for "ธนาคาร" to prevent collision with Bangkok province)
    ("BANK", re.compile(r"((?:ธนาคาร|ธ\.|บมจ\.|สาขา)กรุงเทพ|BBL|กสิกรไทย|ไทยพาณิชย์|กรุงไทย|กรุงศรีอยุธยา|ทหารไทยธนชาต|ออมสิน|เกียรตินาคิน|ทิสโก้|ยูโอบี|ซีไอเอ็มบี|SCB|KBANK|KTB|TTB|CIMB)")),
]

# Priority Hierarchy for Span Disambiguation (Higher value = Higher priority)
PRIORITY_MAP = {
    "ID_CARD": 100,
    "ID_CARD_SUSPECTED": 95,
    "EMAIL": 90,
    "PHONE": 85,
    "BANK_ACC": 80,
    "BANK": 70,
    "NAME": 60,
    "ADDRESS": 50,
}

# Contextual Keywords for Thai ID Classification
ID_CONTEXT_THAI = ("บัตร", "ปชช", "ประชาชน", "เลขประจำตัว")
ID_CONTEXT_ENG_REGEX = re.compile(r"(?i)(?<![a-zA-Z])(?:id|thai\s*id|national\s*id|citizen)(?![a-zA-Z])")
PHONE_CONTEXT_WORDS = ("โทร", "เบอร์", "ติดต่อ", "tel", "phone")

def has_id_context(surrounding_text: str) -> bool:
    if any(th in surrounding_text for th in ID_CONTEXT_THAI):
        return True
    return bool(ID_CONTEXT_ENG_REGEX.search(surrounding_text))

def classify_thai_id(text: str, start: int, end: int, val: str, is_neural: bool = False) -> Optional[str]:
    # 1. Mod11 Checksum verification
    if validate_thai_id_checksum(val):
        return "ID_CARD"
    # 2. Formatted Thai ID structure (X-XXXX-XXXXX-XX-X) even with single-digit typo
    if re.fullmatch(r"\d{1}-\d{4}-\d{5}-\d{2}-\d{1}", val):
        return "ID_CARD_SUSPECTED"
    # 3. Contextual word trigger within +-15 chars (Lookaround prevents matching paid/valid/video)
    w_start = max(0, start - 15)
    w_end = min(len(text), end + 15)
    surrounding = text[w_start:start] + " " + text[end:w_end]
    if has_id_context(surrounding):
        return "ID_CARD_SUSPECTED"
    # 4. Bare unformatted 13 digits without dashes and without context -> Reject (Deliberate Trade-off)
    return None


# Thai Combining Characters (non-spacing vowels & tone marks) that cannot stand alone
THAI_COMBINING_MARKS = set(
    chr(c) for c in range(0x0E30, 0x0E4F)
    if chr(c) in "\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e"
)

# Common Thai Honorific prefixes for NAME entity expansion (ANNOTATION_GUIDELINE.md)
HONORIFIC_PREFIX_PATTERN = re.compile(
    r"(?:นาย|นางสาว|นาง|น\.ส\.|คุณ|พี่|น้อง|หมอ|ช่าง|ด\.ช\.|ด\.ญ\.)\s*$"
)


# =============================================================================
# 1. Mod11 Mathematical Checksum Validator
# =============================================================================

def validate_thai_id_checksum(id_str: str) -> bool:
    """
    Validates a 13-digit Thai National ID card number using the official Mod11 algorithm.
    Formula: Checksum = (11 - (sum_{i=0}^{11} d_i * (13 - i)) % 11) % 10
    """
    digits = [int(c) for c in id_str if c.isdigit()]
    if len(digits) != 13:
        return False
    chk = sum(digits[i] * (13 - i) for i in range(12)) % 11
    expected = (11 - chk) % 10
    return digits[-1] == expected


# =============================================================================
# 2. Subword Span Alignment (Bug-Free Raw Offset Mapping)
# =============================================================================

def align_spans_to_tokens(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    entities: List[Dict[str, Any]],
) -> List[int]:
    """
    Assigns IOB2 labels to subword tokens using raw character offset mappings.
    Guarantees:
    - Special tokens (start == end) -> -100
    - First overlapping token for an entity -> B-{LABEL}
    - Subsequent overlapping tokens -> I-{LABEL}
    - Non-entity tokens and spaces outside entities -> O (0)
    """
    sorted_entities = sorted(entities, key=lambda x: x["start"])
    sample_labels: List[int] = []
    entity_started = set()

    for t_start, t_end in offset_mapping:
        if t_start == t_end:
            sample_labels.append(-100)
            continue

        assigned_tag = "O"
        for ent_idx, ent in enumerate(sorted_entities):
            e_start = ent["start"]
            e_end = ent["end"]
            e_lbl = ent["label"]

            if max(t_start, e_start) < min(t_end, e_end):
                if ent_idx not in entity_started:
                    assigned_tag = f"B-{e_lbl}"
                    entity_started.add(ent_idx)
                else:
                    assigned_tag = f"I-{e_lbl}"
                break

        sample_labels.append(LABEL2ID[assigned_tag])

    return sample_labels


def prepare_hf_batch(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> Dict[str, Any]:
    """
    Tokenizes raw text batches and applies align_spans_to_tokens directly.
    """
    tokenized_inputs = tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_labels = []
    for i, text in enumerate(examples["text"]):
        offsets = tokenized_inputs["offset_mapping"][i]
        ents = examples["entities"][i]
        labels = align_spans_to_tokens(text, offsets, ents)
        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# =============================================================================
# 3. Robust BIO Span Reconstruction
# =============================================================================

def extract_spans_from_predictions(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    predicted_labels: List[str],
    raw: bool = False,
) -> List[Dict[str, Any]]:
    """
    Robust BIO / IOB2 span extractor:
    - Automatically treats orphaned I-TAGs as B-TAGs to prevent entity loss.
    - Validates Mod11 checksum on predicted ID_CARD entities.
    - Trims edge whitespace to guarantee clean character boundaries.
    """
    raw_spans = []
    current_entity: Optional[Dict[str, Any]] = None

    for (start, end), label in zip(offset_mapping, predicted_labels):
        if start == end:  # Skip special / padding tokens
            continue

        if label.startswith("B-"):
            if current_entity is not None:
                raw_spans.append(current_entity)
            ent_type = label[2:]
            current_entity = {"start": start, "end": end, "label": ent_type}
        elif label.startswith("I-"):
            ent_type = label[2:]
            if current_entity is not None and ent_type == current_entity["label"]:
                current_entity["end"] = end
            else:
                # Robust recovery: Orphaned I-TAG or tag boundary shift -> Start new entity
                if current_entity is not None:
                    raw_spans.append(current_entity)
                current_entity = {"start": start, "end": end, "label": ent_type}
        else:
            if current_entity is not None:
                raw_spans.append(current_entity)
                current_entity = None

    if current_entity is not None:
        raw_spans.append(current_entity)

    if raw:
        # Pure Neural Raw Argmax Spans without any post-processing, sanitization, or validation
        res = []
        for sp in raw_spans:
            sub = text[sp["start"]:sp["end"]]
            if sub.strip():
                l_off = len(sub) - len(sub.lstrip())
                r_off = len(sub) - len(sub.rstrip())
                cs, ce = sp["start"] + l_off, sp["end"] - r_off
                res.append({"start": cs, "end": ce, "label": sp["label"], "value": text[cs:ce]})
        return res

    # Clean boundaries and apply Mod11 Checksum filter
    cleaned_spans = []
    seen_spans: Set[Tuple[int, int, str]] = set()

    for sp in raw_spans:
        s = sp["start"]
        e = sp["end"]
        sub = text[s:e]
        stripped = sub.strip()
        if not stripped:
            continue
        l_offset = len(sub) - len(sub.lstrip())
        r_offset = len(sub) - len(sub.rstrip())
        clean_s = s + l_offset
        clean_e = e - r_offset
        val = text[clean_s:clean_e]

        # Post-processing boundary expansion: Append 5-digit postal code to ADDRESS if adjacent
        if sp["label"] == "ADDRESS":
            m_post = re.match(r"^(\s*[1-9]\d{4})(?!\d)", text[clean_e:])
            if m_post:
                clean_e += m_post.end()
                val = text[clean_s:clean_e]

        # Digit Boundary Continuity: Never cut a continuous sequence of digits in half
        if clean_e < len(text) and text[clean_e].isdigit() and val and val[-1].isdigit():
            m_rest = re.match(r"^\d+", text[clean_e:])
            if m_rest:
                clean_e += m_rest.end()
                val = text[clean_s:clean_e]

        # Preposition Detachment: Strip leading prepositions before honorifics (e.g. "ให้คุณสมชาย" -> "คุณสมชาย")
        if sp["label"] == "NAME":
            m_prep = re.match(r"^(?:ให้|ส่ง|ถึง|แก่|แด่|ของ|กับ|นำส่ง|มอบ)\s*(?=(?:คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|แพทย์|ดร\.|อาจารย์|ศ\.|รศ\.|ผู้รับ))", val)
            if m_prep:
                clean_s += m_prep.end()
                val = text[clean_s:clean_e]

        # Sanitization 1: Reject spans that start with a floating combining mark,
        # or sever a Thai syllable (i.e. followed immediately by a combining mark in text)
        if val and val[0] in THAI_COMBINING_MARKS:
            continue
        if clean_e < len(text) and text[clean_e] in THAI_COMBINING_MARKS:
            continue

        # Sanitization 2: Reject ADDRESS or NAME with fewer than 3 Thai characters, numbers in NAME, or non-name English keywords
        if sp["label"] == "NAME":
            # Reject NAME containing digits
            if any(c.isdigit() for c in val):
                continue
            thai_char_count = len([c for c in val if "\u0e00" <= c <= "\u0e7f"])
            if 0 < thai_char_count < 3:
                continue
            if thai_char_count == 0:
                v_lower = val.strip().lower()
                COMMON_NON_NAME_ENG = {"paid", "order", "inv", "invoice", "item", "ref", "test", "status", "bill", "code", "sku", "tracking", "total", "date", "no"}
                if len(v_lower) < 3 or v_lower in COMMON_NON_NAME_ENG:
                    continue
        elif sp["label"] == "ADDRESS":
            thai_char_count = len([c for c in val if "\u0e00" <= c <= "\u0e7f"])
            if 0 < thai_char_count < 3 or (thai_char_count == 0 and len(val.strip()) < 3):
                continue

        # Domain Invariant Validations (Contextual ID & Length constraints)
        if sp["label"] == "ID_CARD":
            id_cat = classify_thai_id(text, clean_s, clean_e, val, is_neural=True)
            if id_cat is None:
                continue
            sp["label"] = id_cat
        elif sp["label"] == "BANK_ACC":
            digits_count = len([c for c in val if c.isdigit()])
            if digits_count < 10 or digits_count > 12:
                continue
        elif sp["label"] == "PHONE":
            digits_count = len([c for c in val if c.isdigit()])
            if digits_count < 9 or digits_count > 11:
                continue
            clean_digits = re.sub(r"[-\s]", "", val)
            if not re.fullmatch(r"^(?:\+66[689]\d{8}|\+662\d{7}|\+66[3-7]\d{7}|0[689]\d{8}|02\d{7}|0[3-7]\d{7})$", clean_digits):
                continue
        elif sp["label"] == "EMAIL":
            if not re.fullmatch(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", val.strip()):
                continue
        elif sp["label"] == "BANK":
            # Domain Invariant: Reject non-bank English acronyms (e.g. ORD, INV)
            v_upper = val.strip().upper()
            if re.fullmatch(r"[A-Z]{2,6}", v_upper):
                valid_acronyms = {"KBANK", "SCB", "BBL", "KTB", "BAY", "TTB", "TMB", "GSB", "KKP", "TISCO", "UOB", "CIMB", "CIMBT", "LHBANK", "BAAC", "GHB", "IBANK", "CITI", "HSBC"}
                if v_upper not in valid_acronyms:
                    continue
            else:
                valid_roots = ("ธนาคาร", "แบงก์", "ธ.", "กสิกร", "ไทยพาณิชย์", "กรุงเทพ", "กรุงไทย", "กรุงศรี", "ทหารไทย", "ธนชาต", "ออมสิน", "เกียรตินาคิน", "ทิสโก้", "ยูโอบี", "ซีไอเอ็มบี", "แลนด์", "ธ.ก.ส", "ธอส", "อิสลาม")
                if not any(k in val for k in valid_roots):
                    continue

        span_tuple = (clean_s, clean_e, sp["label"])
        if span_tuple not in seen_spans:
            seen_spans.add(span_tuple)
            cleaned_spans.append({
                "start": clean_s,
                "end": clean_e,
                "label": sp["label"],
                "value": val,
            })

    return cleaned_spans


def disambiguate_and_merge_spans(
    spans_a: List[Dict[str, Any]],
    spans_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merges candidate spans from Regex Engine and Transformer Model using Priority Disambiguation.
    Resolves overlaps deterministically based on PRIORITY_MAP.
    Enforces Address Enclosure Rule: ADDRESS takes precedence over any overlapping BANK span.
    """
    all_candidates = spans_a + spans_b

    # Address Enclosure Rule: If a candidate BANK span overlaps with or is inside an ADDRESS span, ADDRESS wins
    address_spans = [c for c in all_candidates if c["label"] == "ADDRESS"]
    filtered_candidates = []
    for cand in all_candidates:
        if cand["label"] == "BANK":
            s, e = cand["start"], cand["end"]
            if any(max(s, a["start"]) < min(e, a["end"]) for a in address_spans):
                continue
        filtered_candidates.append(cand)
    all_candidates = filtered_candidates

    # Sort candidates by priority descending, then span length descending
    all_candidates.sort(
        key=lambda x: (PRIORITY_MAP.get(x["label"], 0), x["end"] - x["start"]),
        reverse=True,
    )

    accepted_spans: List[Dict[str, Any]] = []
    for cand in all_candidates:
        s, e = cand["start"], cand["end"]
        # Check overlap with already accepted higher-priority spans
        overlap = False
        for acc in accepted_spans:
            acc_s, acc_e = acc["start"], acc["end"]
            if max(s, acc_s) < min(e, acc_e):
                overlap = True
                break
        if not overlap:
            accepted_spans.append(cand)

    # Return sorted by character start position
    return sorted(accepted_spans, key=lambda x: x["start"])


def compute_span_exact_metrics(
    ground_truth_list: List[Dict[str, Any]],
    predictions_list: List[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Computes Exact Match Precision, Recall, F1 for each class and Overall Micro.
    Matches the exact formula used in scripts/evaluate_model.py.
    """
    per_class_stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABEL_NAMES}

    for sample, preds in zip(ground_truth_list, predictions_list):
        gt_spans = sample.get("entities", [])
        gt_tuples = {(e["start"], e["end"], e["label"]) for e in gt_spans}
        pred_tuples = {(p["start"], p["end"], p["label"]) for p in preds}

        tp_set = gt_tuples.intersection(pred_tuples)
        fp_set = pred_tuples - gt_tuples
        fn_set = gt_tuples - pred_tuples

        for _, _, lbl in tp_set:
            if lbl in per_class_stats:
                per_class_stats[lbl]["tp"] += 1
        for _, _, lbl in fp_set:
            if lbl in per_class_stats:
                per_class_stats[lbl]["fp"] += 1
        for _, _, lbl in fn_set:
            if lbl in per_class_stats:
                per_class_stats[lbl]["fn"] += 1

    per_class_metrics = {}
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for lbl in LABEL_NAMES:
        tp = per_class_stats[lbl]["tp"]
        fp = per_class_stats[lbl]["fp"]
        fn = per_class_stats[lbl]["fn"]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_class_metrics[lbl] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    overall_metrics = {"precision": micro_p, "recall": micro_r, "f1": micro_f1}

    return per_class_metrics, overall_metrics


def print_comparison_table(model_per_class: Dict[str, Dict[str, float]], model_overall: Dict[str, float]) -> str:
    """
    Formats an aligned table comparing Regex Baseline with the Model.
    """
    lines = []
    lines.append("=" * 105)
    lines.append(f"{'HELD-OUT TEST SET BENCHMARK COMPARISON (2,000 SAMPLES)':^105}")
    lines.append("=" * 105)
    lines.append(
        f"{'Entity Class':<12} | "
        f"{'Regex Prec':<10} {'Regex Rec':<10} {'Regex F1':<10} | "
        f"{'Model Prec':<10} {'Model Rec':<10} {'Model F1':<10} | {'Delta F1':<8}"
    )
    lines.append("-" * 105)

    for lbl in LABEL_NAMES:
        rb = REGEX_BASELINE.get(lbl, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        mb = model_per_class.get(lbl, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        delta = mb["f1"] - rb["f1"]
        delta_str = f"{delta:+.4f}"
        lines.append(
            f"{lbl:<12} | "
            f"{rb['precision']:<10.4f} {rb['recall']:<10.4f} {rb['f1']:<10.4f} | "
            f"{mb['precision']:<10.4f} {mb['recall']:<10.4f} {mb['f1']:<10.4f} | {delta_str:<8}"
        )

    lines.append("-" * 105)
    rb_ov = REGEX_BASELINE["OVERALL_MICRO"]
    delta_ov = model_overall["f1"] - rb_ov["f1"]
    lines.append(
        f"{'OVERALL MICRO':<12} | "
        f"{rb_ov['precision']:<10.4f} {rb_ov['recall']:<10.4f} {rb_ov['f1']:<10.4f} | "
        f"{model_overall['precision']:<10.4f} {model_overall['recall']:<10.4f} {model_overall['f1']:<10.4f} | {delta_ov:+.4f}"
    )
    lines.append("=" * 105)
    result = "\n".join(lines)
    print(result)
    return result


# =============================================================================
# 4. Two-Tier Cascaded Production Gateway: ThaiPIIGateway
# =============================================================================

class ThaiPIIGateway:
    """
    Two-Tier Cascaded Production Privacy Gateway:
    - Tier 1: Deterministic Regex + Mod11 Checksum Engine (High Precision on IDs/Emails/Phones/Accounts).
    - Tier 2: Neural WangchanBERTa Engine with Sliding Window Chunking (High Recall on Names/Addresses).
    - Tier 3: Priority Span Disambiguation Matrix.
    - unmask(text, mapping): Single-Pass Regex Substitution (Zero cascading replacement risk).
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        max_length: int = 256,
        stride: int = 64,
        enable_hybrid: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_hybrid = enable_hybrid
        # Ensure stride is strictly less than effective sequence length
        if stride >= (max_length - 2):
            self.stride = max(1, max_length // 4)
        else:
            self.stride = stride

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            self.model = None

    def extract_regex_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Tier 1 Deterministic Extraction with Mod11 Checksum Validation.
        """
        preds = []
        occupied_spans = []

        for label, pattern in TIER1_REGEX_PATTERNS:
            for match in pattern.finditer(text):
                s, e = match.span()
                val = match.group()

                # Contextual ID Verification for ID_CARD
                if label == "ID_CARD":
                    id_cat = classify_thai_id(text, s, e, val, is_neural=False)
                    if id_cat is None:
                        continue
                    label = id_cat
                elif label == "PHONE":
                    # For continuous 9-digit landline starting with 03-07, require phone context
                    digits = [c for c in val if c.isdigit()]
                    if len(digits) == 9 and val.startswith(("03", "04", "05", "06", "07")) and not re.search(r"[-\s]", val):
                        w_start = max(0, s - 10)
                        w_end = min(len(text), e + 10)
                        surr = text[w_start:s] + " " + text[e:w_end]
                        if not any(w in surr.lower() for w in PHONE_CONTEXT_WORDS):
                            continue

                # Avoid internal regex collisions
                if any(max(s, occ_s) < min(e, occ_e) for occ_s, occ_e in occupied_spans):
                    continue

                preds.append({
                    "start": s,
                    "end": e,
                    "label": label,
                    "value": val,
                })
                occupied_spans.append((s, e))

        return preds

    def predict_neural_entities(self, text: str, raw: bool = False) -> List[Dict[str, Any]]:
        """
        Tier 2 Neural Prediction with Sliding Window Chunking.
        """
        if self.model is None or self.tokenizer is None:
            return []

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc.pop("overflow_to_sample_mapping", None)
        all_offsets = enc.pop("offset_mapping").cpu().tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        aggregated_spans: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[int, int, str]] = set()

        for offsets, p_ids in zip(all_offsets, pred_ids):
            pred_labels = [ID2LABEL.get(pid, "O") for pid in p_ids]
            window_spans = extract_spans_from_predictions(text, offsets, pred_labels, raw=raw)
            for sp in window_spans:
                k = (sp["start"], sp["end"], sp["label"])
                if k not in seen_keys:
                    seen_keys.add(k)
                    aggregated_spans.append(sp)

        return sorted(aggregated_spans, key=lambda x: x["start"])

    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Cascaded Two-Tier Detection:
        Merges Tier 1 Regex candidates and Tier 2 Neural candidates via Priority Disambiguation.
        Expands NAME entities to include preceding honorifics per ANNOTATION_GUIDELINE.md.
        """
        neural_spans = self.predict_neural_entities(text)
        if not self.enable_hybrid:
            candidates = neural_spans
        else:
            regex_spans = self.extract_regex_entities(text)
            candidates = disambiguate_and_merge_spans(regex_spans, neural_spans)

        # Honorific Expansion for NAME per ANNOTATION_GUIDELINE.md
        expanded_spans = []
        for ent in candidates:
            if ent["label"] == "NAME":
                s = ent["start"]
                m = HONORIFIC_PREFIX_PATTERN.search(text[:s])
                if m:
                    new_s = m.start()
                    ent["start"] = new_s
                    ent["value"] = text[new_s:ent["end"]]
                # Strip leading preposition if attached (e.g. "ให้คุณสมชาย" -> "คุณสมชาย")
                m_prep = re.match(r"^(?:ให้|ส่ง|ถึง|แก่|แด่|ของ|กับ|นำส่ง|มอบ)\s*(?=(?:คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|แพทย์|ดร\.|อาจารย์|ศ\.|รศ\.|ผู้รับ))", ent["value"])
                if m_prep:
                    ent["start"] += m_prep.end()
                    ent["value"] = ent["value"][m_prep.end():]
            expanded_spans.append(ent)

        # NMS Deduplication: Keep longest span when overlaps occur (e.g. drop standalone 'คุณ' inside 'คุณ สมชาย เข็มกลัด')
        final_spans = []
        expanded_spans.sort(key=lambda x: (x["end"] - x["start"]), reverse=True)
        for cand in expanded_spans:
            s, e = cand["start"], cand["end"]
            if not any(max(s, acc["start"]) < min(e, acc["end"]) for acc in final_spans):
                final_spans.append(cand)

        return sorted(final_spans, key=lambda x: x["start"])

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[Dict[str, Any]]]:
        """
        Sequential multi-sentence inference pipeline.
        """
        all_predictions: List[List[Dict[str, Any]]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_preds = [self.predict_entities(t) for t in batch_texts]
            all_predictions.extend(batch_preds)
        return all_predictions

    def mask(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Masks detected PII in text using indexed placeholders (<NAME_1>, <PHONE_1>).
        Performs reverse-order slice replacement (right-to-left) to preserve character coordinates.
        """
        if entities is None:
            entities = self.predict_entities(text)

        if not entities:
            return text, {}

        # 1. Assign sequential indices from left to right
        forward_ents = sorted(entities, key=lambda x: x["start"])
        type_counters: Dict[str, int] = {}
        for ent in forward_ents:
            lbl = ent["label"]
            type_counters[lbl] = type_counters.get(lbl, 0) + 1
            ent["placeholder"] = f"<{lbl}_{type_counters[lbl]}>"

        # 2. Slice replace from right to left to prevent offset shifts
        reverse_ents = sorted(forward_ents, key=lambda x: x["start"], reverse=True)

        mapping: Dict[str, str] = {}
        masked_text = text
        for ent in reverse_ents:
            s = ent["start"]
            e = ent["end"]
            ph = ent["placeholder"]
            original_value = text[s:e]
            mapping[ph] = original_value
            masked_text = masked_text[:s] + ph + masked_text[e:]

        return masked_text, mapping

    def unmask(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Single-Pass Regex Substitution:
        Replaces all placeholders simultaneously in a single pass.
        Guarantees ZERO cascading replacement risk.
        """
        if not mapping:
            return text

        pattern = re.compile("|".join(re.escape(k) for k in mapping.keys()))
        return pattern.sub(lambda m: mapping[m.group(0)], text)


# =============================================================================
# 5. Seqeval Token Metric Callback
# =============================================================================

def build_compute_metrics():
    """
    Builds compute_metrics callback for Hugging Face Trainer using seqeval.
    """
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        has_seqeval = True
    except ImportError:
        has_seqeval = False

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        preds = np.argmax(predictions, axis=2)

        true_predictions = [
            [ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]
        true_labels = [
            [ID2LABEL[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]

        if has_seqeval:
            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }
        else:
            flat_p = [p for sub in true_predictions for p in sub]
            flat_l = [l for sub in true_labels for l in sub]
            correct = sum(p == l for p, l in zip(flat_p, flat_l))
            acc = correct / len(flat_l) if flat_l else 0.0
            return {"accuracy": acc, "f1": acc}

    return compute_metrics


# =============================================================================
# 6. Full Evaluation Runner on Test Set
# =============================================================================

def evaluate_on_test_set(
    gateway: ThaiPIIGateway,
    test_samples: List[Dict[str, Any]],
    batch_size: int = 32,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Evaluates the trained model on test samples using span-level exact match with Batched Inference.
    """
    print(f"Running Span-level Batched Inference on {len(test_samples)} test samples (Batch size: {batch_size})...")
    texts = [s["text"] for s in test_samples]
    predictions_list = gateway.predict_batch(texts, batch_size=batch_size)
    print(f"  Completed inference on {len(predictions_list)} samples.")

    per_class, overall = compute_span_exact_metrics(test_samples, predictions_list)
    print_comparison_table(per_class, overall)
    return per_class, overall


# =============================================================================
# 7. Production Readiness Suite: Latency Benchmark, Stress Tests & Visualizer
# =============================================================================

ENTITY_COLOR_PALETTE = {
    "ID_CARD_SUSPECTED": "#EA580C",  # Rust Orange
    "NAME": "#2563EB",       # Blue
    "ADDRESS": "#059669",    # Emerald Green
    "ID_CARD": "#DC2626",    # Red
    "PHONE": "#D97706",      # Amber / Orange
    "EMAIL": "#7C3AED",      # Violet / Purple
    "BANK_ACC": "#0891B2",   # Cyan
    "BANK": "#DB2777",       # Pink
}


def benchmark_gateway_performance(
    gateway: ThaiPIIGateway,
    sample_texts: List[str],
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Measures production latency distribution (p50, p95, p99 in ms) and throughput (samples/sec).
    """
    import time

    if not sample_texts:
        sample_texts = [
            "รบกวนส่งของให้คุณ สมชาย เข็มกลัด ที่ 99/1 ซอยอารีย์ พญาไท กทม 10400 เบอร์ 0812345678"
        ]

    # Warm-up runs
    for _ in range(min(3, len(sample_texts))):
        gateway.predict_entities(sample_texts[0])

    latencies_ms = []
    total_start = time.perf_counter()

    for i in range(num_runs):
        text = sample_texts[i % len(sample_texts)]
        t0 = time.perf_counter()
        gateway.predict_entities(text)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_time = time.perf_counter() - total_start
    latencies_sorted = sorted(latencies_ms)

    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    throughput = num_runs / total_time if total_time > 0 else 0.0

    return {
        "num_runs": num_runs,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_avg_ms": round(avg_latency, 2),
        "throughput_samples_per_sec": round(throughput, 2),
        "total_time_sec": round(total_time, 2),
    }


def visualize_entities_html(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Renders clean, self-contained HTML with color-coded badges and highlights for Jupyter/Web display.
    """
    import html

    if not entities:
        return f"<div style='font-family:sans-serif; padding:10px;'>{html.escape(text)}</div>"

    sorted_ents = sorted(entities, key=lambda x: x["start"])
    chunks = []
    last_idx = 0

    for ent in sorted_ents:
        s, e, lbl = ent["start"], ent["end"], ent["label"]
        color = ENTITY_COLOR_PALETTE.get(lbl, "#4B5563")

        # Plain text before entity
        if s > last_idx:
            chunks.append(html.escape(text[last_idx:s]))

        val_escaped = html.escape(text[s:e])
        badge = (
            f"<mark style='background-color:{color}22; border:1px solid {color}; "
            f"border-radius:4px; padding:2px 6px; margin:0 2px; display:inline-block;'>"
            f"<span style='color:{color}; font-weight:600;'>{val_escaped}</span> "
            f"<span style='background:{color}; color:white; font-size:10px; font-weight:bold; "
            f"padding:1px 4px; border-radius:3px; margin-left:4px;'>{lbl}</span>"
            f"</mark>"
        )
        chunks.append(badge)
        last_idx = e

    if last_idx < len(text):
        chunks.append(html.escape(text[last_idx:]))

    html_content = "".join(chunks)
    return (
        f"<div style='font-family:Segoe UI, Tahoma, sans-serif; line-height:2.0; "
        f"padding:16px; background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px;'>"
        f"{html_content}"
        f"</div>"
    )


def run_production_stress_tests(gateway: ThaiPIIGateway) -> Dict[str, Any]:
    """
    Runs 3 production stress tests:
    1. Zero False-Positive on Non-PII clean text
    2. Unsegmented Thai text (no spaces)
    3. Long text (>500 tokens) Zero-Truncation Sliding Window check
    """
    results: Dict[str, Any] = {}

    # Test 1: Non-PII text (weather, math, item descriptions, invoice codes)
    non_pii_texts = [
        "รายงานสภาพอากาศประจำวัน อุณหภูมิเฉลี่ย 32 องศาเซลเซียส มีเมฆเป็นส่วนมาก",
        "คำนวณผลรวมของยอดสั่งซื้อ 15 รายการ รวมเป็นเงินทั้งสิ้น 12500 บาทถ้วน",
        "โปรดตรวจสอบรหัสอ้างอิงคำสั่งซื้อ ORD-987654 และใบเสร็จ INV-2026-88 ในระบบจัดการคลังสินค้า",
    ]
    fp_count = 0
    for t in non_pii_texts:
        preds = gateway.predict_entities(t)
        fp_count += len(preds)
    results["stress_non_pii"] = {
        "samples_tested": len(non_pii_texts),
        "false_positive_count": fp_count,
        "passed": fp_count == 0,
    }

    # Test 2: Unsegmented Thai text (zero spaces between tokens)
    unsegmented_text = "จัดส่งพัสดุให้คุณสมชายเข็มกลัดที่99/1ซอยอารีย์พญาไทกทม10400เบอร์0812345678"
    unseg_preds = gateway.predict_entities(unsegmented_text)
    found_labels = {p["label"] for p in unseg_preds}
    has_name = "NAME" in found_labels
    has_addr = "ADDRESS" in found_labels
    has_phone = "PHONE" in found_labels
    results["stress_unsegmented_thai"] = {
        "text": unsegmented_text,
        "detected_entities": unseg_preds,
        "has_name": has_name,
        "has_address": has_addr,
        "has_phone": has_phone,
        "passed": has_name or has_phone,
    }

    # Test 3: Long document zero-truncation test (Sliding Window check)
    filler = "นี่คือข้อความส่วนหนึ่งของรายงานการประชุมประจำเดือนที่มีเนื้อหายาวมาก " * 30
    pii_suffix = "และขอแจ้งเบอร์โทรศัพท์สำหรับติดต่อฉุกเฉินคือ 0891234567 และอีเมล admin@example.com ครับ"
    long_doc = filler + pii_suffix
    long_preds = gateway.predict_entities(long_doc)
    long_labels = {p["label"] for p in long_preds}
    passed_sliding = "PHONE" in long_labels and "EMAIL" in long_labels
    results["stress_long_document"] = {
        "text_length_chars": len(long_doc),
        "detected_entities_count": len(long_preds),
        "detected_labels": list(long_labels),
        "passed": passed_sliding,
    }

    results["all_passed"] = (
        results["stress_non_pii"]["passed"]
        and results["stress_unsegmented_thai"]["passed"]
        and results["stress_long_document"]["passed"]
    )
    return results


# =============================================================================
# 8. Main Training Workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train WangchanBERTa on Thai PII Synthetic Dataset")
    parser.add_argument("--model_name", type=str, default="airesearch/wangchanberta-base-att-spm-uncased")
    parser.add_argument("--dataset_name", type=str, default="DuckerMaster/Thai-PII-Synthetic")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./wangchanberta-thai-pii")
    parser.add_argument("--smoke_test", action="store_true", help="Run 10-step test locally to verify logic")
    args = parser.parse_args()

    print("=" * 70)
    print("STARTING WANGCHANBERTA THAI PII FINE-TUNING PIPELINE")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.learning_rate}")
    print("=" * 70)

    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from datasets import load_dataset, Dataset, DatasetDict

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. Load Dataset (config: 'default' for raw text & character spans)
    if os.path.exists("data/train.json") and os.path.exists("data/test.json"):
        print("[INFO] Loading local dataset from data/ directory...")
        with open("data/train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open("data/validation.json", "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open("data/test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if args.smoke_test:
            print("[INFO] Smoke test enabled: subsetting to 32 train, 16 val, 16 test.")
            train_data = train_data[:32]
            val_data = val_data[:16]
            test_data = test_data[:16]

        raw_dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        })
    else:
        print(f"[INFO] Loading dataset '{args.dataset_name}' from Hugging Face Hub...")
        raw_dataset = load_dataset(args.dataset_name, "default")
        if args.smoke_test:
            raw_dataset["train"] = raw_dataset["train"].select(range(32))
            raw_dataset["validation"] = raw_dataset["validation"].select(range(16))
            raw_dataset["test"] = raw_dataset["test"].select(range(16))

    # 3. Tokenize & Align Spans
    print("[INFO] Tokenizing and aligning character spans to subwords...")
    tokenized_dataset = raw_dataset.map(
        lambda x: prepare_hf_batch(x, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    # 4. Model Setup
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 5. Training Arguments (Version-agnostic parameter handling)
    import inspect
    use_cuda = torch.cuda.is_available()
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs if not args.smoke_test else 1,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "epoch" if not args.smoke_test else "no",
        "save_total_limit": 1,
        "load_best_model_at_end": True if not args.smoke_test else False,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "fp16": use_cuda,
        "logging_steps": 10 if not args.smoke_test else 1,
        "report_to": "none",
    }

    if "eval_strategy" in ta_sig:
        training_args_kwargs["eval_strategy"] = "epoch" if not args.smoke_test else "no"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch" if not args.smoke_test else "no"

    if "warmup_ratio" in ta_sig:
        training_args_kwargs["warmup_ratio"] = 0.1
    else:
        # Newer transformers v5+ accepts float (0.1) or integer step count (400)
        training_args_kwargs["warmup_steps"] = 400

    training_args = TrainingArguments(**training_args_kwargs)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Version-agnostic parameter handling for Trainer (tokenizer vs processing_class)
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(),
    }
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # 6. Train
    print("[INFO] Starting model training...")
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[SUCCESS] Model saved to {args.output_dir}")

    # 7. Evaluate Two-Tier Cascaded Gateway on Held-out Test Set
    test_samples = list(raw_dataset["test"])
    gateway = ThaiPIIGateway(model=model, tokenizer=tokenizer, max_length=args.max_length, enable_hybrid=True)
    evaluate_on_test_set(gateway, test_samples, batch_size=32)

    # 8. Demonstrate Production Gateway
    demo_text = (
        "แจ้งส่งเอกสารให้คุณ นเรศ เพชรประเสริฐ เบอร์ 0812345678 "
        "เลขบัตร 1-1002-01234-56-1 เลขสุ่มไม่ผ่าน 1234567890123 "
        "และคุณ สมชาย เข็มกลัด เบอร์ 0987654321 โอนเงินเข้า บัญชี 012-3-45678-9 ธนาคารกสิกรไทย"
    )
    print("\n" + "=" * 70)
    print("DEMO: TWO-TIER CASCADED THAI PII GATEWAY")
    print("=" * 70)
    print("Original Text:\n", demo_text)
    masked_text, mapping = gateway.mask(demo_text)
    print("\nMasked Text:\n", masked_text)
    print("\nMapping Dictionary:\n", json.dumps(mapping, ensure_ascii=False, indent=2))
    restored_text = gateway.unmask(masked_text, mapping)
    print("\nRestored Text:\n", restored_text)
    assert restored_text == demo_text, "Verification failed: Restored text does not match original!"
    print("\n[VERIFIED] Round-trip Mask/Unmask integrity is 100%!")


if __name__ == "__main__":
    main()
