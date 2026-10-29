"""
Unit & Invariant Tests using pytest (tests/test_dataset_invariants.py)
Validates dataset integrity against project invariants.
"""

import os
import json
import re
import pytest

try:
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False

@pytest.fixture(scope="module")
def dataset_splits():
    splits = {}
    for s in ["train", "validation", "test"]:
        p = os.path.join("data", f"{s}.json")
        assert os.path.exists(p), f"Missing split file: {p}"
        with open(p, "r", encoding="utf-8") as f:
            splits[s] = json.load(f)
    return splits

def test_span_integrity(dataset_splits):
    """Every entity span must exactly match the character slice."""
    for s, data in dataset_splits.items():
        for item in data:
            text = item["text"]
            for ent in item["entities"]:
                st, en, val = ent["start"], ent["end"], ent["value"]
                assert 0 <= st < en <= len(text)
                assert text[st:en] == val

def test_no_span_overlap(dataset_splits):
    """Entities within a single sample must not overlap."""
    for s, data in dataset_splits.items():
        for item in data:
            ents = sorted(item["entities"], key=lambda x: x["start"])
            for i in range(len(ents) - 1):
                assert ents[i]["end"] <= ents[i+1]["start"]

def test_mod11_checksum(dataset_splits):
    """All generated Thai IDs must pass the mathematical Mod11 Checksum."""
    for s, data in dataset_splits.items():
        for item in data:
            for ent in item["entities"]:
                if ent["label"] == "ID_CARD":
                    digits = [int(c) for c in ent["value"] if c.isdigit()]
                    assert len(digits) == 13
                    chk = sum(digits[i] * (13 - i) for i in range(12)) % 11
                    expected = (11 - chk) % 10
                    assert digits[-1] == expected

def test_schema_fields(dataset_splits):
    """Every record must contain clean required fields without row-level metadata bloat."""
    required_fields = {"id", "text", "entities", "split", "template_id"}
    prohibited_fields = {"generator_version", "seed", "language", "license"}
    
    for s, data in dataset_splits.items():
        for item in data:
            assert set(item.keys()) == required_fields, f"Record keys mismatch: {set(item.keys())}"
            assert item["split"] == s
            assert prohibited_fields.isdisjoint(set(item.keys())), f"Metadata bloat found in record: {item['id']}"

    # Verify dataset-level metadata in manifest
    with open(os.path.join("data", "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["language"] == "th"
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["generator_version"].startswith(("v5.", "v6."))

def test_split_leakage_templates(dataset_splits):
    """Zero template overlap between Train and Test."""
    tr_tmpls = set(x["template_id"] for x in dataset_splits["train"])
    te_tmpls = set(x["template_id"] for x in dataset_splits["test"])
    overlap = tr_tmpls.intersection(te_tmpls)
    assert len(overlap) == 0

def test_split_leakage_names(dataset_splits):
    """Zero lexical Name overlap between Train and Test."""
    def get_names(records):
        names = set()
        for r in records:
            for e in r["entities"]:
                if e["label"] == "NAME":
                    v = e["value"]
                    for t in ["นาย", "นางสาว", "นาง", "น.ส.", "คุณ", "พี่", "น้อง", "ช่าง", "หมอ", "Mr. ", "Ms. ", "K. "]:
                        if v.startswith(t):
                            v = v[len(t):].strip()
                            break
                    names.add(v)
        return names

    tr_names = get_names(dataset_splits["train"])
    te_names = get_names(dataset_splits["test"])
    overlap = tr_names.intersection(te_names)
    assert len(overlap) == 0

def test_address_density(dataset_splits):
    """ADDRESS density must be >= 20.0% in all splits."""
    for s, data in dataset_splits.items():
        total_ents = sum(len(x["entities"]) for x in data)
        addr_cnt = sum(1 for x in data for e in x["entities"] if e["label"] == "ADDRESS")
        pct = (addr_cnt / total_ents * 100) if total_ents > 0 else 0
        assert pct >= 20.0

@pytest.mark.skipif(not PYTHAINLP_AVAILABLE, reason="PyThaiNLP is not installed")
def test_tokenizer_boundary_alignment(dataset_splits):
    """Entities must align with PyThaiNLP token boundaries."""
    for s, data in dataset_splits.items():
        for item in data:
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
                assert ent["start"] in starts
                assert ent["end"] in ends

def test_annotation_boundary_rules(dataset_splits):
    """Enforce strict Annotation Boundary Rules:
    1. Addresses ending near a postal code must include the 5-digit code inside the ADDRESS span.
    2. Names preceded by recognized honorifics must include the honorific inside the NAME span.
    """
    honorifics = ["นาย", "นางสาว", "นาง", "น.ส.", "คุณ", "พี่", "น้อง", "ช่าง", "หมอ"]
    for s, data in dataset_splits.items():
        for item in data:
            text = item["text"]
            for ent in item["entities"]:
                if ent["label"] == "ADDRESS":
                    trailing = text[ent["end"]:]
                    postal_match = re.search(r'^\s*(\d{5})\b', trailing)
                    assert not postal_match, (
                        f"Postal code '{postal_match.group(1)}' left outside ADDRESS in [{s}:{item['id']}]"
                    )
                elif ent["label"] == "NAME":
                    preceding = text[:ent["start"]].rstrip()
                    for h in honorifics:
                        assert not preceding.endswith(h), (
                            f"Honorific '{h}' left outside NAME in [{s}:{item['id']}]"
                        )

def test_hf_token_classification_export():
    """Validates that Hugging Face IOB2 export meets formatting standards."""
    export_dir = os.path.join("data", "hf_export")
    assert os.path.exists(export_dir), "hf_export directory missing"
    
    # Check label mappings
    assert os.path.exists("config/label2id.json"), "label2id.json missing"
    assert os.path.exists("config/id2label.json"), "id2label.json missing"
    with open("config/label2id.json", encoding="utf-8") as f:
        l2id = json.load(f)
    assert len(l2id) == 15, f"Expected 15 classes in label2id, got {len(l2id)}"
    assert l2id["O"] == 0

    for s in ["train", "validation", "test"]:
        p = os.path.join(export_dir, f"{s}.json")
        assert os.path.exists(p), f"Exported file {p} missing"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0
        for item in data:  # Full verification across all 100% samples
            assert "tokens" in item and "ner_tags" in item and "ner_labels" in item
            assert len(item["tokens"]) == len(item["ner_tags"]) == len(item["ner_labels"])
            assert "".join(item["tokens"]) == item["text"]
            for tag in item["ner_tags"]:
                assert 0 <= tag < 15
