"""
Unit & Integration Tests for WangchanBERTa Pipeline & Two-Tier Cascaded ThaiPIIGateway
(tests/test_wangchanberta_pipeline.py)
"""

import os
import re
import sys
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from transformers import AutoTokenizer

from scripts.train_wangchanberta_pii import (
    align_spans_to_tokens,
    extract_spans_from_predictions,
    compute_span_exact_metrics,
    print_comparison_table,
    validate_thai_id_checksum,
    disambiguate_and_merge_spans,
    ThaiPIIGateway,
    benchmark_gateway_performance,
    visualize_entities_html,
    run_production_stress_tests,
    LABEL2ID,
    ID2LABEL,
    LABEL_NAMES,
)


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class MockTransformerModel(torch.nn.Module):
    """Mock PyTorch module returning realistic logits for integration testing."""
    def __init__(self, target_label_idx: int = 1):
        super().__init__()
        self.target_label_idx = target_label_idx

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, len(LABEL2ID)), dtype=torch.float32)
        logits[:, :, 0] = 5.0  # Default to "O"
        if seq_len > 9:
            logits[:, 5:10, 0] = 0.0
            logits[:, 5:10, self.target_label_idx] = 10.0
        return DummyOutput(logits)


# =============================================================================
# 1. Mod11 Checksum Validator Tests
# =============================================================================

def test_validate_thai_id_checksum():
    """
    Verifies that the Mod11 algorithm correctly identifies valid and invalid Thai IDs.
    """
    # Valid Thai IDs with and without dashes
    assert validate_thai_id_checksum("1-1002-01234-56-1") is True
    assert validate_thai_id_checksum("1100201234561") is True

    # Invalid Thai IDs (wrong length or checksum mismatch)
    assert validate_thai_id_checksum("1234567890123") is False
    assert validate_thai_id_checksum("0000000000000") is False
    assert validate_thai_id_checksum("12345") is False


# =============================================================================
# 2. Token Alignment & BIO Recovery Tests
# =============================================================================

def test_align_spans_to_tokens_accuracy():
    text = "แจ้งคุณ นเรศ เพชรประเสริฐ เบอร์ 0812345678"
    offsets: List[Tuple[int, int]] = [
        (0, 0), (0, 1), (0, 4), (4, 7), (7, 8),
        (8, 9), (9, 12), (12, 13), (13, 17), (17, 25),
        (25, 26), (26, 31), (31, 32),
        (32, 34), (34, 36), (36, 38), (38, 40), (40, 42), (0, 0),
    ]
    entities = [
        {"start": 8, "end": 25, "label": "NAME"},
        {"start": 32, "end": 42, "label": "PHONE"},
    ]

    labels = align_spans_to_tokens(text, offsets, entities)
    assert labels[0] == -100
    assert labels[-1] == -100
    assert labels[4] == LABEL2ID["O"]       # Leading space
    assert labels[5] == LABEL2ID["B-NAME"]  # First subword
    assert labels[6] == LABEL2ID["I-NAME"]  # Next subword
    assert labels[13] == LABEL2ID["B-PHONE"]


def test_extract_spans_orphaned_i_tag_recovery():
    """
    Verifies that an orphaned I-TAG without a preceding B-TAG is treated as a B-TAG (Robust BIO).
    """
    text = "ส่งให้ สมชาย เข็มกลัด"
    offsets = [(0, 6), (7, 13), (14, 22)]
    pred_labels = ["O", "I-NAME", "I-NAME"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    assert len(spans) == 1
    assert spans[0]["label"] == "NAME"
    assert spans[0]["start"] == 7
    assert spans[0]["end"] == 22
    assert spans[0]["value"] == "สมชาย เข็มกลัด"


# =============================================================================
# 3. Two-Tier Cascaded Gateway & Disambiguation Tests
# =============================================================================

def test_mod11_filter_in_gateway():
    """
    Verifies that invalid 13-digit numbers are NOT masked as ID_CARD.
    """
    gateway = ThaiPIIGateway(enable_hybrid=True)
    text = "บัตรประชาชน 1-1002-01234-56-1 และเลขออเดอร์ 1234567890123"

    masked, mapping = gateway.mask(text)
    # The valid ID should be masked
    assert "1-1002-01234-56-1" not in masked
    assert "<ID_CARD_1>" in masked
    # The invalid 13 digits must NOT be masked as ID_CARD
    assert "1234567890123" in masked


def test_priority_overlap_resolution():
    """
    Verifies that high-priority regex entity (PHONE: priority 85)
    wins over lower-priority overlapping entity (ADDRESS: priority 50).
    """
    spans_regex = [{"start": 10, "end": 20, "label": "PHONE", "value": "0812345678"}]
    spans_neural = [{"start": 5, "end": 25, "label": "ADDRESS", "value": "ที่อยู่ 0812345678 กทม"}]

    merged = disambiguate_and_merge_spans(spans_regex, spans_neural)
    assert len(merged) == 1
    assert merged[0]["label"] == "PHONE"
    assert merged[0]["value"] == "0812345678"


def test_single_pass_unmask_cascading_safety():
    """
    Adversarial test: One entity value contains another placeholder's literal string.
    Single-pass substitution must NOT cause cascading replacement.
    """
    gateway = ThaiPIIGateway()
    # Mapping where <NAME_1> resolves to a string containing <PHONE_1>
    adversarial_mapping = {
        "<NAME_1>": "<PHONE_1>",
        "<PHONE_1>": "0812345678",
    }
    masked_text = "สวัสดี <NAME_1> และ <PHONE_1>"
    restored = gateway.unmask(masked_text, adversarial_mapping)

    # In single-pass: <NAME_1> becomes "<PHONE_1>" and <PHONE_1> becomes "0812345678" simultaneously
    assert restored == "สวัสดี <PHONE_1> และ 0812345678"


def test_bank_address_no_collision():
    """
    Verifies that 'กทม 10400' or 'กรุงเทพ 10400' in addresses does NOT trigger BANK regex.
    Also verifies that Address Enclosure Rule protects ADDRESS from BANK overlap.
    """
    gateway = ThaiPIIGateway(enable_hybrid=True)
    text = "จัดส่งพัสดุไปที่ 99/1 ซอยอารีย์ แขวงพญาไท เขตพญาไท กรุงเทพ 10400"
    regex_entities = gateway.extract_regex_entities(text)
    # Tier 1 Regex must NOT detect bare 'กรุงเทพ' as BANK
    bank_matches = [e for e in regex_entities if e["label"] == "BANK"]
    assert len(bank_matches) == 0

    # Address Enclosure Rule: if neural predicted ADDRESS contains 'กรุงเทพ' and a spurious BANK span exists,
    # ADDRESS must take precedence over BANK.
    fake_bank = [{"start": 48, "end": 55, "label": "BANK", "value": "กรุงเทพ"}]
    fake_address = [{"start": 17, "end": 61, "label": "ADDRESS", "value": "99/1 ซอยอารีย์ แขวงพญาไท เขตพญาไท กรุงเทพ 10400"}]
    merged = disambiguate_and_merge_spans(fake_bank, fake_address)
    assert len(merged) == 1
    assert merged[0]["label"] == "ADDRESS"
    assert merged[0]["value"] == "99/1 ซอยอารีย์ แขวงพญาไท เขตพญาไท กรุงเทพ 10400"


def test_bank_explicit_detection():
    """
    Verifies that explicit bank mentions (ธนาคารกรุงเทพ, บมจ.กรุงเทพ, BBL) ARE detected as BANK.
    """
    gateway = ThaiPIIGateway(enable_hybrid=True)
    text = "กรุณาโอนเงินเข้า ธนาคารกรุงเทพ หรือ BBL เลขที่ 123-4-56789-0"
    regex_entities = gateway.extract_regex_entities(text)
    bank_matches = [e for e in regex_entities if e["label"] == "BANK"]
    assert len(bank_matches) == 2
    assert bank_matches[0]["value"] == "ธนาคารกรุงเทพ"
    assert bank_matches[1]["value"] == "BBL"


def test_address_postal_code_expansion():
    """
    Verifies that a 5-digit postal code immediately following an ADDRESS span
    is automatically merged into the ADDRESS boundary by extract_spans_from_predictions.
    """
    text = "ที่อยู่ 99/1 ซอยอารีย์ พญาไท กทม 10400 ครับ"
    # Neural model predicted ADDRESS up to 'กทม' (offset 7 to 32)
    offsets = [(0, 6), (7, 32), (32, 38), (38, 43)]
    pred_labels = ["O", "B-ADDRESS", "O", "O"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    assert len(spans) == 1
    assert spans[0]["label"] == "ADDRESS"
    assert spans[0]["value"] == "99/1 ซอยอารีย์ พญาไท กทม 10400"
    assert spans[0]["end"] == 38


def test_syllable_severing_rejected():
    """
    Verifies that a span severing a Thai syllable (e.g. 'ท' from 'ที่' with trailing combining mark)
    is rejected by extract_spans_from_predictions.
    """
    text = "จัดส่งไป ที่ 99/1 ซอยอารีย์"
    # 'ท' in 'ที่' is at index 10..11, followed by 'ี' at 11
    offsets = [(0, 9), (10, 11), (11, 13)]
    pred_labels = ["O", "B-ADDRESS", "O"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    # Must reject the severed 'ท'
    assert len(spans) == 0


def test_short_thai_name_address_rejected():
    """
    Verifies that spurious 1 or 2 character Thai spans are rejected for NAME and ADDRESS.
    """
    text = "ส่งให้ ก และ ข ที่ ค"
    offsets = [(7, 8), (13, 14), (19, 20)]
    pred_labels = ["B-NAME", "B-NAME", "B-ADDRESS"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    assert len(spans) == 0


def test_honorific_name_expansion():
    """
    Verifies that if a model predicts only 'สมชาย เข็มกลัด', the Gateway automatically
    expands the NAME entity to include preceding honorifics (e.g. 'คุณ').
    """
    gateway = ThaiPIIGateway()
    text = "รบกวนส่งของให้คุณ สมชาย เข็มกลัด ที่ 99/1 ซอยอารีย์"
    # Mock neural entities containing only the name without 'คุณ'
    s = text.find("สมชาย เข็มกลัด")
    e = s + len("สมชาย เข็มกลัด")
    neural_spans = [{"start": s, "end": e, "label": "NAME", "value": "สมชาย เข็มกลัด"}]

    gateway.predict_neural_entities = lambda t: neural_spans
    entities = gateway.predict_entities(text)

    name_ents = [ent for ent in entities if ent["label"] == "NAME"]
    assert len(name_ents) == 1
    assert name_ents[0]["value"] == "คุณ สมชาย เข็มกลัด"
    assert name_ents[0]["start"] == text.find("คุณ")


def test_no_floating_vowels_in_masked_text():
    """
    Verifies that masked text contains no severed syllables or floating vowel/tone marks.
    """
    gateway = ThaiPIIGateway()
    text = "รบกวนส่งของให้คุณ สมชาย เข็มกลัด ที่ 99/1 ซอยอารีย์ พญาไท กทม 10400"
    entities = [
        {"start": text.find("คุณ สมชาย เข็มกลัด"), "end": text.find("คุณ สมชาย เข็มกลัด") + len("คุณ สมชาย เข็มกลัด"), "label": "NAME", "value": "คุณ สมชาย เข็มกลัด"},
        {"start": text.find("99/1 ซอยอารีย์ พญาไท กทม 10400"), "end": text.find("99/1 ซอยอารีย์ พญาไท กทม 10400") + len("99/1 ซอยอารีย์ พญาไท กทม 10400"), "label": "ADDRESS", "value": "99/1 ซอยอารีย์ พญาไท กทม 10400"}
    ]
    masked, mapping = gateway.mask(text, entities=entities)
    assert not re.search(r">\s*[่-๋ะ-ู]", masked)
    assert "<ADDRESS_1>ี่" not in masked


def test_gateway_mask_and_unmask_roundtrip():
    """
    Verifies round-trip mask & unmask integrity.
    """
    gateway = ThaiPIIGateway(enable_hybrid=True)
    text = "ติดต่อ คุณสมชาย เข็มกลัด หรือ คุณสมหญิง วงศ์สวัสดิ์ โทร 0812345678 และ 0987654321"
    entities = [
        {"start": 10, "end": 24, "label": "NAME", "value": "สมชาย เข็มกลัด"},
        {"start": 33, "end": 51, "label": "NAME", "value": "สมหญิง วงศ์สวัสดิ์"},
        {"start": 56, "end": 66, "label": "PHONE", "value": "0812345678"},
        {"start": 71, "end": 81, "label": "PHONE", "value": "0987654321"},
    ]

    masked_text, mapping = gateway.mask(text, entities=entities)
    assert "<NAME_1>" in masked_text
    assert "<NAME_2>" in masked_text
    assert "<PHONE_1>" in masked_text
    assert "<PHONE_2>" in masked_text

    restored_text = gateway.unmask(masked_text, mapping)
    assert restored_text == text


# =============================================================================
# 4. Neural Pipeline & Batch Integration Tests
# =============================================================================

def test_gateway_predict_entities_integration():
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    mock_model = MockTransformerModel(target_label_idx=1)
    gateway = ThaiPIIGateway(model=mock_model, tokenizer=tokenizer, device="cpu", max_length=64, enable_hybrid=True)

    text = "ขอแจ้งข้อมูลคุณ นเรศ เพชรประเสริฐ เบอร์โทรศัพท์ 0812345678 ครับ"
    spans = gateway.predict_entities(text)
    assert isinstance(spans, list)
    assert len(spans) > 0


def test_gateway_sliding_window_zero_truncation():
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    mock_model = MockTransformerModel(target_label_idx=1)
    gateway = ThaiPIIGateway(model=mock_model, tokenizer=tokenizer, device="cpu", max_length=32, stride=8, enable_hybrid=False)

    long_text = "ข้อความเนื้อหายาวเริ่มต้นที่ตรงนี้ " * 20
    spans = gateway.predict_entities(long_text)
    assert isinstance(spans, list)


def test_gateway_predict_batch():
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    mock_model = MockTransformerModel(target_label_idx=1)
    gateway = ThaiPIIGateway(model=mock_model, tokenizer=tokenizer, device="cpu", max_length=64, enable_hybrid=True)

    batch_texts = ["แจ้งคุณ นเรศ เบอร์ 0812345678", "แจ้งคุณ สมชาย เบอร์ 0891234567"]
    batch_preds = gateway.predict_batch(batch_texts, batch_size=2)
    assert len(batch_preds) == 2


def test_compute_span_exact_metrics_and_table():
    gt = [
        {"entities": [{"start": 0, "end": 5, "label": "NAME"}]},
        {"entities": [{"start": 10, "end": 20, "label": "PHONE"}]},
    ]
    pred = [
        [{"start": 0, "end": 5, "label": "NAME"}],
        [{"start": 10, "end": 20, "label": "PHONE"}],
    ]

    per_class, overall = compute_span_exact_metrics(gt, pred)
    assert per_class["NAME"]["f1"] == 1.0
    assert per_class["PHONE"]["f1"] == 1.0
    assert overall["f1"] == 1.0

    table_str = print_comparison_table(per_class, overall)
    assert "HELD-OUT TEST SET BENCHMARK COMPARISON" in table_str
    assert "OVERALL MICRO" in table_str


# =============================================================================
# 5. Production Readiness Suite Tests
# =============================================================================

def test_benchmark_gateway_performance():
    gateway = ThaiPIIGateway(enable_hybrid=True)
    texts = [
        "แจ้งส่งเอกสารให้คุณ นเรศ เพชรประเสริฐ เบอร์ 0812345678",
        "โอนเงินเข้า บัญชี 012-3-45678-9 ธนาคารกสิกรไทย",
    ]
    perf = benchmark_gateway_performance(gateway, texts, num_runs=10)
    assert "latency_p50_ms" in perf
    assert "latency_p95_ms" in perf
    assert "throughput_samples_per_sec" in perf
    assert perf["latency_p50_ms"] >= 0.0
    assert perf["throughput_samples_per_sec"] > 0.0


def test_visualize_entities_html():
    text = "ส่งของให้คุณ สมชาย ที่ 99/1 ซอยอารีย์"
    entities = [
        {"start": 10, "end": 18, "label": "NAME", "value": "สมชาย"},
        {"start": 23, "end": 37, "label": "ADDRESS", "value": "99/1 ซอยอารีย์"},
    ]
    html_out = visualize_entities_html(text, entities)
    assert "<mark" in html_out
    assert "NAME" in html_out
    assert "ADDRESS" in html_out
    assert "สมชาย" in html_out
    assert "#2563EB" in html_out


def test_production_stress_suite():
    gateway = ThaiPIIGateway(enable_hybrid=True)
    results = run_production_stress_tests(gateway)
    assert "stress_non_pii" in results
    assert "stress_unsegmented_thai" in results
    assert "stress_long_document" in results
    assert results["stress_non_pii"]["passed"] is True
    assert results["stress_long_document"]["passed"] is True


def test_bank_domain_invariant_rejection():
    """
    Verifies that arbitrary English acronyms like 'ORD' or 'INV' predicted as BANK
    are rejected by domain invariant validator, while legitimate banks are kept.
    """
    text = "รหัสอ้างอิง ORD-123 และ INV-456 โอนผ่าน SCB"
    offsets = [(12, 15), (24, 27), (40, 43)]
    pred_labels = ["B-BANK", "B-BANK", "B-BANK"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    # Only SCB should survive
    assert len(spans) == 1
    assert spans[0]["value"] == "SCB"
    assert spans[0]["label"] == "BANK"


def test_unsegmented_preposition_detachment():
    """
    Verifies that leading prepositions ('ให้', 'ส่ง', 'ถึง') in unsegmented Thai text
    are detached from NAME entity when followed by an honorific ('คุณ', 'นาย', etc.).
    """
    text = "รบกวนส่งของให้คุณสมชายเข็มกลัดที่99/1ซอยอารีย์พญาไทกทม10400เบอร์0812345678"
    offsets = [(11, 22), (30, 59), (64, 74)]
    pred_labels = ["B-NAME", "B-ADDRESS", "B-PHONE"]

    spans = extract_spans_from_predictions(text, offsets, pred_labels)
    name_spans = [s for s in spans if s["label"] == "NAME"]
    assert len(name_spans) == 1
    # Must detach 'ให้' and preserve exactly 'คุณสมชาย'
    assert name_spans[0]["value"] == "คุณสมชาย"
    assert not name_spans[0]["value"].startswith("ให้")


def test_regression_11_cases():
    """
    Unit test covering the 11 controlled regression cases at regex / deterministic gateway level.
    """
    gateway = ThaiPIIGateway(enable_hybrid=True)

    # 1. ภาษาไทยติดตัวเลข + บริบทบัตร
    res1 = gateway.extract_regex_entities("บัตร1234567890123")
    assert any(e["label"] == "ID_CARD_SUSPECTED" and e["value"] == "1234567890123" for e in res1)

    # 2. เลข 13 หลักมีขีด แม้ Checksum ผิด
    res2 = gateway.extract_regex_entities("1-1002-01234-56-0")
    assert any(e["label"] == "ID_CARD_SUSPECTED" and e["value"] == "1-1002-01234-56-0" for e in res2)

    # 3. เบอร์บ้านมีขีด
    res3 = gateway.extract_regex_entities("02-123-4567")
    assert any(e["label"] == "PHONE" and e["value"] == "02-123-4567" for e in res3)

    # 4. เบอร์บ้าน 9 หลักติดกัน (02)
    res4 = gateway.extract_regex_entities("021234567")
    assert any(e["label"] == "PHONE" and e["value"] == "021234567" for e in res4)

    # 5. เลข 9 หลักขึ้นต้น 0 ไม่มีบริบท -> ต้องไม่จับเป็นเบอร์
    res5 = gateway.extract_regex_entities("012345678")
    assert len(res5) == 0

    # 6. เบอร์มือถือ 10 หลัก
    res6 = gateway.extract_regex_entities("0812345678")
    assert any(e["label"] == "PHONE" and e["value"] == "0812345678" for e in res6)

    # 7. เลขบัญชีติดกับภาษาไทย
    res7 = gateway.extract_regex_entities("บัญชี1234567890")
    assert any(e["label"] == "BANK_ACC" and e["value"] == "1234567890" for e in res7)

    # 8. คำว่า paid ต้องไม่ trigger id
    res8 = gateway.extract_regex_entities("paid1234567890123")
    assert not any(e["label"] in ("ID_CARD", "ID_CARD_SUSPECTED") for e in res8)

    # 9. รหัสคำสั่งซื้อ (Deliberate Trade-off on Regex)
    res9 = gateway.extract_regex_entities("ORD-1234567890123")
    assert not any(e["label"] in ("ID_CARD", "ID_CARD_SUSPECTED") for e in res9)

    # 10. เลขสุ่ม 13 หลักลอยๆ ไม่มีขีด ไม่มีบริบท (Deliberate Trade-off on Regex)
    res10 = gateway.extract_regex_entities("เลขสุ่ม 1234567890123")
    assert not any(e["label"] in ("ID_CARD", "ID_CARD_SUSPECTED") for e in res10)

    # 11. เลขบัตรประชาชนจริง Mod11 Checksum ถูกต้อง
    res11 = gateway.extract_regex_entities("เลขบัตร 1-1002-01234-56-1")
    assert any(e["label"] == "ID_CARD" and e["value"] == "1-1002-01234-56-1" for e in res11)


def test_id_card_suspected_mask_unmask_roundtrip():
    gateway = ThaiPIIGateway(enable_hybrid=True)
    text = "กรุณาตรวจสอบ บัตร1234567890123 และ 1-1002-01234-56-0"
    masked, mapping = gateway.mask(text)
    assert "<ID_CARD_SUSPECTED_1>" in masked
    assert "<ID_CARD_SUSPECTED_2>" in masked
    restored = gateway.unmask(masked, mapping)
    assert restored == text



def test_neural_phone_prefix_sanitization():
    """
    Verifies that neural predictions of invalid phone numbers (e.g. 012345678)
    are strictly rejected by post-processing, while valid ones (0812345678) are preserved.
    """
    from scripts.train_wangchanberta_pii import extract_spans_from_predictions

    # Invalid Thai phone prefix (01)
    spans_invalid = extract_spans_from_predictions("012345678", [(0, 9)], ["B-PHONE"])
    assert len(spans_invalid) == 0, "012345678 must be rejected by post-processing"

    # Valid Thai mobile prefix (08)
    spans_valid = extract_spans_from_predictions("0812345678", [(0, 10)], ["B-PHONE"])
    assert len(spans_valid) == 1 and spans_valid[0]["label"] == "PHONE"

    # Valid Bangkok landline prefix (02)
    spans_bkk = extract_spans_from_predictions("021234567", [(0, 9)], ["B-PHONE"])
    assert len(spans_bkk) == 1 and spans_bkk[0]["label"] == "PHONE"


def test_neural_email_sanitization():
    from scripts.train_wangchanberta_pii import extract_spans_from_predictions
    # Fake/garbage text predicted as EMAIL must be rejected
    spans_invalid = extract_spans_from_predictions('paid1234567890123', [(0, 17)], ['B-EMAIL'])
    assert len(spans_invalid) == 0, 'paid1234567890123 must not be classified as EMAIL'

    # Real email must pass
    spans_valid = extract_spans_from_predictions('user@example.com', [(0, 16)], ['B-EMAIL'])
    assert len(spans_valid) == 1 and spans_valid[0]['label'] == 'EMAIL'


def test_neural_name_sanitization():
    from scripts.train_wangchanberta_pii import extract_spans_from_predictions
    # Names with digits must be rejected
    assert len(extract_spans_from_predictions('paid1234567890123', [(0, 17)], ['B-NAME'])) == 0
    # Common English non-names must be rejected
    assert len(extract_spans_from_predictions('paid1234567890123', [(0, 4)], ['B-NAME'])) == 0
    # Real names must pass
    assert len(extract_spans_from_predictions('สมชาย เข็มกลัด', [(0, 14)], ['B-NAME'])) == 1
    assert len(extract_spans_from_predictions('John Doe', [(0, 8)], ['B-NAME'])) == 1
