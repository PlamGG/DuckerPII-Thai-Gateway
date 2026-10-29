"""
Hugging Face Token Classification Exporter (scripts/export_hf_token_classification.py)
Converts character span annotations into standard IOB2 token classification format:
- tokens: List[str] (PyThaiNLP newmm tokenization)
- ner_tags: List[int] (integer IDs 0..14)
- ner_labels: List[str] (IOB2 string tags: B-NAME, I-NAME, O, etc.)
- entities: List[Dict] (original character span metadata preserved)

Exports to: data/hf_export/{train, validation, test}.json
Also exports config/label2id.json and config/id2label.json
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple

try:
    from pythainlp.tokenize import word_tokenize
except ImportError:
    print("[ERROR] pythainlp is required. Run 'pip install pythainlp'")
    sys.exit(1)

LABELS = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]

def generate_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {"O": 0}
    id2label = {0: "O"}
    for idx, lbl in enumerate(LABELS):
        b_tag = f"B-{lbl}"
        i_tag = f"I-{lbl}"
        b_id = 2 * idx + 1
        i_id = 2 * idx + 2
        label2id[b_tag] = b_id
        label2id[i_tag] = i_id
        id2label[b_id] = b_tag
        id2label[i_id] = i_tag
    return label2id, id2label

def convert_sample(item: Dict[str, Any], label2id: Dict[str, int]) -> Dict[str, Any]:
    text = item["text"]
    entities = sorted(item.get("entities", []), key=lambda x: x["start"])
    tokens = word_tokenize(text, engine="newmm")

    # Compute token character offsets
    token_spans = []
    cur = 0
    for tok in tokens:
        token_spans.append((cur, cur + len(tok)))
        cur += len(tok)

    # Reconstructed text check
    assert "".join(tokens) == text, f"Token reconstruction mismatch in {item['id']}"

    ner_labels = []
    ner_tags = []

    for st, en in token_spans:
        assigned_tag = "O"
        for ent in entities:
            e_st = ent["start"]
            e_en = ent["end"]
            e_lbl = ent["label"]
            
            # Check if this token falls within the entity span
            if st >= e_st and en <= e_en:
                if st == e_st:
                    assigned_tag = f"B-{e_lbl}"
                else:
                    assigned_tag = f"I-{e_lbl}"
                break
        
        ner_labels.append(assigned_tag)
        ner_tags.append(label2id[assigned_tag])

    assert len(tokens) == len(ner_tags) == len(ner_labels)

    return {
        "id": item["id"],
        "text": text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "ner_labels": ner_labels,
        "entities": entities,
        "split": item.get("split", ""),
        "template_id": item.get("template_id", "")
    }

def run_export():
    print("=====================================================")
    print("RUNNING HUGGING FACE TOKEN CLASSIFICATION EXPORTER")
    print("=====================================================")

    label2id, id2label = generate_label_mappings()

    os.makedirs("config", exist_ok=True)
    with open("config/label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open("config/id2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    print(f"[*] Generated label mappings: {len(label2id)} classes (Saved to config/)")

    export_dir = os.path.join("data", "hf_export")
    os.makedirs(export_dir, exist_ok=True)

    splits = ["train", "validation", "test"]
    stats = {}

    for s in splits:
        src_path = os.path.join("data", f"{s}.json")
        if not os.path.exists(src_path):
            print(f"[ERROR] Source file not found: {src_path}")
            sys.exit(1)

        with open(src_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        converted_data = []
        total_tokens = 0
        total_entity_tokens = 0

        for item in raw_data:
            conv = convert_sample(item, label2id)
            converted_data.append(conv)
            total_tokens += len(conv["tokens"])
            total_entity_tokens += sum(1 for tag in conv["ner_tags"] if tag != 0)

        out_path = os.path.join(export_dir, f"{s}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)

        stats[s] = {
            "samples": len(converted_data),
            "total_tokens": total_tokens,
            "entity_tokens": total_entity_tokens,
            "out_file": out_path
        }
        print(f" - {s:<10}: {len(converted_data):>5} samples | {total_tokens:>7} tokens | {total_entity_tokens:>6} entity tokens -> {out_path}")

    print("=====================================================")
    print("EXPORT COMPLETED SUCCESSFULLY (EXIT CODE 0)")
    print("=====================================================")
    return stats

if __name__ == "__main__":
    run_export()
