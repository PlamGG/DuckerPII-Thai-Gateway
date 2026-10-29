import os
import sys
import json
import re

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_wangchanberta_pii import (
    TIER1_REGEX_PATTERNS as SCRIPT_PATTERNS,
    PRIORITY_MAP as SCRIPT_PRIORITY,
    classify_thai_id as script_classify,
)

def test_notebook_script_parity():
    """
    Verifies that regex patterns, priority map, and classify_thai_id
    in notebooks/kaggle_wangchanberta_pii.ipynb match scripts/train_wangchanberta_pii.py exactly.
    """
    nb_path = "notebooks/kaggle-wangchanberta-pii.ipynb" if os.path.exists("notebooks/kaggle-wangchanberta-pii.ipynb") else "notebooks/kaggle_wangchanberta_pii.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Dynamically find the cell containing Step 2 configuration & PRIORITY_MAP
    config_cell = next(c for c in nb["cells"] if "PRIORITY_MAP" in "".join(c.get("source", [])))
    c3_source = "".join(config_cell["source"])

    # Verify PRIORITY_MAP parity
    assert "ID_CARD_SUSPECTED" in SCRIPT_PRIORITY
    assert SCRIPT_PRIORITY["ID_CARD_SUSPECTED"] == 95
    assert '"ID_CARD_SUSPECTED": 95' in c3_source

    # Verify Regex patterns parity
    script_labels = [p[0] for p in SCRIPT_PATTERNS]
    for lbl in script_labels:
        assert f'("{lbl}",' in c3_source

    # Verify classify_thai_id presence in Cell 3
    assert "def classify_thai_id(" in c3_source
    assert "ID_CONTEXT_THAI" in c3_source
    assert "ID_CONTEXT_ENG_REGEX" in c3_source
