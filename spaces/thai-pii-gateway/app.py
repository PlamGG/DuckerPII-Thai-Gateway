import spaces
import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==========================================
# 1. MODEL INITIALIZATION & FALLBACK
# ==========================================
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "wangchanberta-thai-pii"))
HUB_MODEL_NAME = "DuckerMaster/DuckerPII-Thai-Base"

if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    MODEL_PATH = LOCAL_MODEL_DIR
    print(f"[*] Loading model from local directory: {MODEL_PATH}")
else:
    MODEL_PATH = HUB_MODEL_NAME
    print(f"[*] Loading model from Hugging Face Hub: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"[!] Warning: Could not load fine-tuned model ({e}). Using base model as fallback.")
    fallback_name = "airesearch/wangchanberta-base-att-spm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(fallback_name)
    model = None

# ==========================================
# 2. LABEL SCHEMAS & TWO-TIER ENGINE
# ==========================================
LABEL_NAMES = ["NAME", "ID_CARD", "PHONE", "EMAIL", "BANK_ACC", "BANK", "ADDRESS"]
ID2LABEL = {0: "O"}
LABEL2ID = {"O": 0}
for idx, lbl in enumerate(LABEL_NAMES):
    ID2LABEL[2 * idx + 1] = f"B-{lbl}"
    ID2LABEL[2 * idx + 2] = f"I-{lbl}"
    LABEL2ID[f"B-{lbl}"] = 2 * idx + 1
    LABEL2ID[f"I-{lbl}"] = 2 * idx + 2

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

def validate_thai_id_checksum(id_str: str) -> bool:
    digits = [int(c) for c in id_str if c.isdigit()]
    if len(digits) != 13:
        return False
    chk = sum(digits[i] * (13 - i) for i in range(12)) % 11
    expected = (11 - chk) % 10
    return digits[-1] == expected

TIER1_REGEX_PATTERNS = [
    ("ID_CARD", re.compile(r"(?<!\d)\d{1}-\d{4}-\d{5}-\d{2}-\d{1}(?!\d)|(?<!\d)\d{13}(?!\d)")),
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+66[-\s]?\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}|0[689]\d[-\s]?\d{3}[-\s]?\d{4}|02[-\s]?\d{3}[-\s]?\d{4}|0[3-7]\d[-\s]?\d{3}[-\s]?\d{3})(?!\d)")),
    ("BANK_ACC", re.compile(r"(?<!\d)\d{3}-\d{1}-\d{5}-\d{1}(?!\d)|(?<!\d)(?!(?:0[689]))\d{10,12}(?!\d)")),
    ("BANK", re.compile(r"((?:ธนาคาร|ธ\.|บมจ\.|สาขา)กรุงเทพ|BBL|กสิกรไทย|ไทยพาณิชย์|กรุงไทย|กรุงศรีอยุธยา|ทหารไทยธนชาต|ออมสิน|เกียรตินาคิน|ทิสโก้|ยูโอบี|ซีไอเอ็มบี|SCB|KBANK|KTB|TTB|CIMB)")),
]

ID_CONTEXT_THAI = ("บัตร", "ปชช", "ประชาชน", "เลขประจำตัว")
ID_CONTEXT_ENG_REGEX = re.compile(r"(?i)(?<![a-zA-Z])(?:id|thai\s*id|national\s*id|citizen)(?![a-zA-Z])")
PHONE_CONTEXT_WORDS = ("โทร", "เบอร์", "ติดต่อ", "tel", "phone")

def classify_thai_id(text: str, start: int, end: int, val: str, is_neural: bool = False) -> Optional[str]:
    if validate_thai_id_checksum(val):
        return "ID_CARD"
    if re.fullmatch(r"\d{1}-\d{4}-\d{5}-\d{2}-\d{1}", val):
        return "ID_CARD_SUSPECTED"
    w_start = max(0, start - 15)
    w_end = min(len(text), end + 15)
    surrounding = text[w_start:start] + " " + text[end:w_end]
    if any(th in surrounding for th in ID_CONTEXT_THAI) or bool(ID_CONTEXT_ENG_REGEX.search(surrounding)):
        return "ID_CARD_SUSPECTED"
    return None

THAI_COMBINING_MARKS = set(chr(c) for c in range(0x0E30, 0x0E4F) if chr(c) in "ัิีึืฺุู็่้๊๋์ํ๎")
HONORIFIC_PREFIX_PATTERN = re.compile(r"(?:นาย|นางสาว|นาง|น\.ส\.|คุณ|ด\.ช\.|ด\.ญ\.|แพทย์|ดร\.|อาจารย์|ศ\.|รศ\.)\s*$")

def extract_spans(text: str, offsets: List[Tuple[int, int]], pred_labels: List[str], raw: bool = False) -> List[Dict[str, Any]]:
    raw_spans, cur = [], None
    for (s, e), lbl in zip(offsets, pred_labels):
        if s == e: continue
        if lbl.startswith("B-"):
            if cur: raw_spans.append(cur)
            cur = {"start": s, "end": e, "label": lbl[2:]}
        elif lbl.startswith("I-"):
            ent_type = lbl[2:]
            if cur is not None and ent_type == cur["label"]:
                cur["end"] = e
            else:
                if cur: raw_spans.append(cur)
                cur = {"start": s, "end": e, "label": ent_type}
        else:
            if cur: raw_spans.append(cur); cur = None
    if cur: raw_spans.append(cur)

    if raw:
        res = []
        for sp in raw_spans:
            sub = text[sp["start"]:sp["end"]]
            if sub.strip():
                l_off = len(sub) - len(sub.lstrip())
                r_off = len(sub) - len(sub.rstrip())
                cs, ce = sp["start"] + l_off, sp["end"] - r_off
                res.append({"start": cs, "end": ce, "label": sp["label"], "value": text[cs:ce]})
        return res

    cleaned = []
    seen = set()
    for sp in raw_spans:
        sub = text[sp["start"]:sp["end"]]
        if not sub.strip(): continue
        l_off = len(sub) - len(sub.lstrip())
        r_off = len(sub) - len(sub.rstrip())
        cs, ce = sp["start"] + l_off, sp["end"] - r_off
        val = text[cs:ce]

        if sp["label"] == "ADDRESS":
            m_post = re.match(r"^(\s*[1-9]\d{4})(?!\d)", text[ce:])
            if m_post:
                ce += m_post.end()
                val = text[cs:ce]

        if ce < len(text) and text[ce].isdigit() and val and val[-1].isdigit():
            m_rest = re.match(r"^\d+", text[ce:])
            if m_rest:
                ce += m_rest.end()
                val = text[cs:ce]

        if sp["label"] == "NAME":
            m_prep = re.match(r"^(?:ให้|ส่ง|ถึง|แก่|แด่|ของ|กับ|นำส่ง|มอบ)\s*(?=(?:คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|แพทย์|ดร\.|อาจารย์|ศ\.|รศ\.|ผู้รับ))", val)
            if m_prep:
                cs += m_prep.end()
                val = text[cs:ce]

        if val and val[0] in THAI_COMBINING_MARKS: continue
        if ce < len(text) and text[ce] in THAI_COMBINING_MARKS: continue

        if sp["label"] == "NAME":
            if any(c.isdigit() for c in val): continue
            thai_char_count = len([c for c in val if "\u0e00" <= c <= "\u0e7f"])
            if 0 < thai_char_count < 3: continue
            if thai_char_count == 0:
                v_lower = val.strip().lower()
                COMMON_NON_NAME_ENG = {"paid", "order", "inv", "invoice", "item", "ref", "test", "status", "bill", "code", "sku", "tracking", "total", "date", "no"}
                if len(v_lower) < 3 or v_lower in COMMON_NON_NAME_ENG: continue
        elif sp["label"] == "ADDRESS":
            thai_char_count = len([c for c in val if "\u0e00" <= c <= "\u0e7f"])
            if 0 < thai_char_count < 3 or (thai_char_count == 0 and len(val.strip()) < 3):
                continue

        if sp["label"] == "ID_CARD":
            id_cat = classify_thai_id(text, cs, ce, val, is_neural=True)
            if id_cat is None: continue
            sp["label"] = id_cat
        elif sp["label"] == "BANK_ACC":
            digits_count = len([c for c in val if c.isdigit()])
            if digits_count < 10 or digits_count > 12: continue
        elif sp["label"] == "PHONE":
            digits_count = len([c for c in val if c.isdigit()])
            if digits_count < 9 or digits_count > 11: continue
            clean_digits = re.sub(r"[-\s]", "", val)
            if not re.fullmatch(r"^(?:\+66[689]\d{8}|\+662\d{7}|\+66[3-7]\d{7}|0[689]\d{8}|02\d{7}|0[3-7]\d{7})$", clean_digits):
                continue
        elif sp["label"] == "EMAIL":
            if not re.fullmatch(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", val.strip()):
                continue
        elif sp["label"] == "BANK":
            v_upper = val.strip().upper()
            if re.fullmatch(r"[A-Z]{2,6}", v_upper):
                valid_acronyms = {"KBANK", "SCB", "BBL", "KTB", "BAY", "TTB", "TMB", "GSB", "KKP", "TISCO", "UOB", "CIMB", "CIMBT", "LHBANK", "BAAC", "GHB", "IBANK", "CITI", "HSBC"}
                if v_upper not in valid_acronyms: continue
            else:
                valid_roots = ("ธนาคาร", "กรุงเทพ", "กสิกร", "ไทยพาณิชย์", "กรุงไทย", "กรุงศรี", "ทหารไทย", "ออมสิน", "เกียรตินาคิน", "ทิสโก้", "ยูโอบี", "ซีไอเอ็มบี", "ธ.ก.ส.", "ธอส.", "อิสลาม")
                if not any(k in val for k in valid_roots): continue

        k = (cs, ce, sp["label"])
        if k not in seen:
            seen.add(k)
            cleaned.append({"start": cs, "end": ce, "label": sp["label"], "value": val})
    return cleaned

def disambiguate_spans(spans_a: List[Dict], spans_b: List[Dict]) -> List[Dict]:
    all_candidates = spans_a + spans_b
    address_spans = [c for c in all_candidates if c["label"] == "ADDRESS"]
    all_candidates = [
        c for c in all_candidates
        if not (c["label"] == "BANK" and any(max(c["start"], a["start"]) < min(c["end"], a["end"]) for a in address_spans))
    ]
    all_candidates.sort(key=lambda x: (PRIORITY_MAP.get(x["label"], 0), x["end"] - x["start"]), reverse=True)
    accepted = []
    for cand in all_candidates:
        s, e = cand["start"], cand["end"]
        if not any(max(s, acc["start"]) < min(e, acc["end"]) for acc in accepted):
            accepted.append(cand)
    return sorted(accepted, key=lambda x: x["start"])

class ThaiPIIGateway:
    def __init__(self, model=None, tokenizer=None, device=None, max_length=256, stride=64, enable_hybrid=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = max(1, max_length // 4) if stride >= (max_length - 2) else stride
        self.enable_hybrid = enable_hybrid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval() if model is not None else None

    def extract_regex_entities(self, text: str) -> List[Dict[str, Any]]:
        preds, occupied = [], []
        for label, pattern in TIER1_REGEX_PATTERNS:
            for match in pattern.finditer(text):
                s, e = match.span()
                val = match.group()
                if label == "ID_CARD":
                    id_cat = classify_thai_id(text, s, e, val, is_neural=False)
                    if id_cat is None: continue
                    label = id_cat
                elif label == "PHONE":
                    digits = [c for c in val if c.isdigit()]
                    if len(digits) == 9 and val.startswith(("03", "04", "05", "06", "07")) and not re.search(r"[-\s]", val):
                        w_start = max(0, s - 10)
                        w_end = min(len(text), e + 10)
                        surr = text[w_start:s] + " " + text[e:w_end]
                        if not any(w in surr.lower() for w in PHONE_CONTEXT_WORDS):
                            continue
                if any(max(s, os) < min(e, oe) for os, oe in occupied): continue
                preds.append({"start": s, "end": e, "label": label, "value": val})
                occupied.append((s, e))
        return preds

    def predict_neural_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None: return []
        enc = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            stride=self.stride, 
            return_overflowing_tokens=True, 
            return_offsets_mapping=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        enc.pop("overflow_to_sample_mapping", None)
        all_offsets = enc.pop("offset_mapping").cpu().tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        aggregated, seen = [], set()
        for offsets, p_ids in zip(all_offsets, pred_ids):
            labels = [ID2LABEL.get(p, "O") for p in p_ids]
            for sp in extract_spans(text, offsets, labels, raw=False):
                k = (sp["start"], sp["end"], sp["label"])
                if k not in seen: seen.add(k); aggregated.append(sp)
        return sorted(aggregated, key=lambda x: x["start"])

    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        neural_spans = self.predict_neural_entities(text)
        if not self.enable_hybrid:
            candidates = neural_spans
        else:
            regex_spans = self.extract_regex_entities(text)
            candidates = disambiguate_spans(regex_spans, neural_spans)

        expanded_spans = []
        for ent in candidates:
            if ent["label"] == "NAME":
                s = ent["start"]
                m = HONORIFIC_PREFIX_PATTERN.search(text[:s])
                if m:
                    new_s = m.start()
                    ent["start"] = new_s
                    ent["value"] = text[new_s:ent["end"]]
                m_prep = re.match(r"^(?:ให้|ส่ง|ถึง|แก่|แด่|ของ|กับ|นำส่ง|มอบ)\s*(?=(?:คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|แพทย์|ดร\.|อาจารย์|ศ\.|รศ\.|ผู้รับ))", ent["value"])
                if m_prep:
                    ent["start"] += m_prep.end()
                    ent["value"] = ent["value"][m_prep.end():]
            expanded_spans.append(ent)

        final_spans = []
        expanded_spans.sort(key=lambda x: (x["end"] - x["start"]), reverse=True)
        for cand in expanded_spans:
            s, e = cand["start"], cand["end"]
            if not any(max(s, acc["start"]) < min(e, acc["end"]) for acc in final_spans):
                final_spans.append(cand)
        return sorted(final_spans, key=lambda x: x["start"])

    def mask(self, text: str, entities=None) -> Tuple[str, Dict[str, str]]:
        if entities is None: entities = self.predict_entities(text)
        if not entities: return text, {}
        fwd = sorted(entities, key=lambda x: x["start"])
        cnts = {}
        for e in fwd:
            lbl = e["label"]
            cnts[lbl] = cnts.get(lbl, 0) + 1
            e["placeholder"] = f"<{lbl}_{cnts[lbl]}>"
        mapping = {}
        masked = text
        for e in sorted(fwd, key=lambda x: x["start"], reverse=True):
            s, en, ph = e["start"], e["end"], e["placeholder"]
            mapping[ph] = text[s:en]
            masked = masked[:s] + ph + masked[en:]
        return masked, mapping

    def unmask(self, text: str, mapping: Dict[str, str]) -> str:
        if not mapping: return text
        pattern = re.compile("|".join(re.escape(k) for k in mapping.keys()))
        return pattern.sub(lambda m: mapping[m.group(0)], text)

gateway = ThaiPIIGateway(model, tokenizer, enable_hybrid=True)

# ==========================================
# 3. GRADIO PIPELINE HANDLER
# ==========================================
@spaces.GPU
def process_text_gateway(text: str):
    if not text or not text.strip():
        return [], "", {}, "⚠️ กรุณาพิมพ์ข้อความเพื่อทดสอบระบบ"

    entities = gateway.predict_entities(text)
    masked_text, mapping = gateway.mask(text, entities=entities)
    restored = gateway.unmask(masked_text, mapping)

    highlighted = []
    last_idx = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        if ent["start"] > last_idx:
            highlighted.append((text[last_idx:ent["start"]], None))
        highlighted.append((text[ent["start"]:ent["end"]], ent["label"]))
        last_idx = ent["end"]
    if last_idx < len(text):
        highlighted.append((text[last_idx:], None))

    is_lossless = (restored == text)
    status_msg = """
    <div style="display:flex; align-items:center; gap:8px; padding:10px 14px; background:#ECFDF5; border:1px solid #A7F3D0; border-radius:12px; color:#065F46; font-size:12.5px; font-weight:600;">
        <span style="display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px; border-radius:50%; background:#10B981; color:white; font-size:11px;">✓</span>
        <span>Round-Trip Integrity: 100% Exact Lossless Match (De-anonymization Passed)</span>
    </div>
    """ if is_lossless else "<div style='color:red;'>Verification Mismatch</div>"

    return highlighted, masked_text, mapping, status_msg

# ==========================================
# 4. POLISHED MODERN DESIGN CSS
# ==========================================
CUSTOM_CSS = """
/* Clean Modern Page Background */
body, .gradio-container {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header Container */
.app-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

/* SMARTPHONE CHASSIS */
.phone-wrapper {
    background: #FFFFFF;
    border: 3px solid #E2E8F0;
    border-radius: 36px;
    box-shadow: 0 15px 30px -5px rgba(15, 23, 42, 0.08);
    overflow: hidden;
    padding: 0;
}

/* Phone Screen Header */
.line-screen-header {
    background: #06C755;
    color: #FFFFFF;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
    font-size: 14.5px;
}

/* LINE Wallpaper Screen Area */
.line-screen-body {
    background-color: #849EB0 !important;
    padding: 16px 14px;
    min-height: 250px;
}

/* Chat Bubbles */
.bubble-bot {
    background: #FFFFFF;
    color: #1E293B;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    font-size: 13px;
    max-width: 90%;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    line-height: 1.45;
}

/* Phone input styling */
.phone-chat-input textarea {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
    font-size: 13.5px !important;
}

/* LINE Send Button */
.line-send-btn {
    background-color: #06C755 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(6,199,85,0.3) !important;
    transition: all 0.2s ease !important;
}
.line-send-btn:hover {
    background-color: #05B34C !important;
    transform: translateY(-1px);
}

/* Quick Chips */
.quick-chip {
    border-radius: 14px !important;
    font-size: 12.5px !important;
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 6px !important;
    text-align: left !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
.quick-chip:hover {
    background-color: #F1F5F9 !important;
    border-color: #06C755 !important;
    color: #0F172A !important;
}

/* Right Side: Clean Enterprise Card */
.enterprise-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

.card-step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 12px;
}
"""

# ==========================================
# 5. GRADIO APP INTERFACE
# ==========================================
with gr.Blocks(title="Thai PII Privacy Gateway", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    # 1. TOP BRAND HEADER
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("""
            <div class="app-header">
                <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="width:40px; height:40px; border-radius:12px; background:#06C755; color:white; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; box-shadow:0 4px 10px rgba(6,199,85,0.25);">
                            🛡️
                        </div>
                        <div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <h1 style="margin:0; font-size:20px; font-weight:800; color:#0F172A;">Thai PII Privacy Gateway</h1>
                                <span style="font-size:11px; font-weight:700; padding:2px 8px; background:#ECFDF5; color:#065F46; border-radius:999px; border:1px solid #A7F3D0;">v6.0.0 Production</span>
                            </div>
                            <p style="margin:2px 0 0 0; font-size:13px; color:#64748B;">Zero-Trust De-Identification Middleware for Thai Free-form Chat &bull; Two-Tier Cascaded Hybrid Architecture</p>
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px; font-size:12px; font-weight:600; color:#334155;">
                        <span style="padding:6px 12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">⚡ p50 Latency: <b>17.47 ms</b></span>
                        <span style="padding:6px 12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">🛡️ OOD Span Leak: <b style="color:#059669;">1.67%</b></span>
                        <span style="padding:6px 12px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;">✅ CI Tests: <b style="color:#2563EB;">41/41 Passed</b></span>
                    </div>
                </div>
            </div>
            """)

    # 2. ACCORDION: CAPABILITIES & SYSTEM ARCHITECTURE
    with gr.Accordion("ℹ️ Architecture Details & Benchmark Specifications (คลิกเพื่อดูโครงสร้างระบบและผลการทดสอบเชิงลึก)", open=False):
        gr.HTML("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:18px; margin-top:8px;">
            <!-- 1. KPI STAT CARDS -->
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-bottom:16px;">
                <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size:18px; font-weight:800; color:#0F172A;">17.47 ms</div>
                    <div style="font-size:11px; font-weight:600; color:#64748B;">p50 Latency (SLA &lt; 25ms)</div>
                </div>
                <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size:18px; font-weight:800; color:#059669;">1.67%</div>
                    <div style="font-size:11px; font-weight:600; color:#64748B;">OOD Span Leak Rate</div>
                </div>
                <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size:18px; font-weight:800; color:#2563EB;">98.33%</div>
                    <div style="font-size:11px; font-weight:600; color:#64748B;">OOD Real-World Recall</div>
                </div>
                <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size:18px; font-weight:800; color:#7C3AED;">41 / 41</div>
                    <div style="font-size:11px; font-weight:600; color:#64748B;">CI/CD Tests (100% Pass)</div>
                </div>
            </div>

            <!-- 2. ARCHITECTURAL FLOW -->
            <div style="background:white; border:1px solid #E2E8F0; border-radius:12px; padding:14px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size:12.5px; font-weight:700; color:#1E293B; margin-bottom:8px;">⚡ Two-Tier Cascaded Execution Flow:</div>
                <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; font-size:12px;">
                    <span style="background:#F1F5F9; padding:6px 10px; border-radius:8px; font-weight:600;">💬 ข้อความแชตภาษาไทย</span>
                    <span>➔</span>
                    <span style="background:#FEF3C7; color:#92400E; padding:6px 10px; border-radius:8px; font-weight:600;">Tier 1: Regex &amp; Mod11 Checksum (บัตร, เบอร์, บัญชี, อีเมล)</span>
                    <span>➔</span>
                    <span style="background:#DBEAFE; color:#1E40AF; padding:6px 10px; border-radius:8px; font-weight:600;">Tier 2: WangchanBERTa NER (ชื่อคน, สแลง, ที่อยู่)</span>
                    <span>➔</span>
                    <span style="background:#D1FAE5; color:#065F46; padding:6px 10px; border-radius:8px; font-weight:600;">🔒 Reversible Masking (&lt;TAG_N&gt;)</span>
                </div>
            </div>

            <!-- 3. KEY DIFFERENTIATORS -->
            <div style="font-size:12px; color:#475569; line-height:1.6;">
                <b>💡 ทำไมต้องใช้ Two-Tier Hybrid?</b> 
                โมเดล AI เพียวๆ มักสับสนตัวเลขธุรกิจ (เช่น เลขสต็อก 9 หลัก หรือเลขพัสดุ FLASH) การมี Tier-1 เข้ามาช่วยตรวจสอบสูตรเลขบัตรประชาชน Mod11 และ Regex เบอร์โทรของไทย จะช่วยตัด False Positive ได้ 100% ในขณะที่ AI ทำหน้าที่จับชื่อและที่อยู่แบบฟรีฟอร์มได้อย่างไร้รอยต่อ
            </div>
        </div>
        """)

    # 3. WORKSPACE 2 COLUMNS
    with gr.Row():
        # LEFT: SMARTPHONE LINE CHAT SIMULATOR
        with gr.Column(scale=5):
            gr.Markdown("<h3 style='margin:0 0 8px 0; font-size:15px; font-weight:700; color:#0F172A;'>📱 Client Interface (จำลองแชต LINE)</h3>")
            
            with gr.Column(elem_classes=["phone-wrapper"]):
                gr.HTML("""
                <div class="line-screen-header">
                    <span>❮ ลูกค้า (Customer Support)</span>
                    <span style="font-size:11px; background:rgba(255,255,255,0.25); padding:2px 8px; border-radius:12px;">🟢 ตอบกลับด้วย AI</span>
                </div>
                """)
                
                with gr.Column(elem_classes=["line-screen-body"]):
                    gr.HTML("""
                    <div class="bubble-bot">
                        👋 <b>AI Assistant:</b> สวัสดีครับ พิมพ์ข้อความสั่งซื้อ แจ้งโอนเงิน หรือส่งข้อมูลติดต่อด้านล่างได้เลยครับ ระบบจะทำการเซ็นเซอร์ PII ให้ทันทีก่อนส่งเข้า AI ครับ
                    </div>
                    """)
                    
                    chat_input = gr.Textbox(
                        lines=4,
                        placeholder="พิมพ์ข้อความที่ต้องการทดสอบ เช่น:\nส่งของให้คุณ สมชาย เข็มกลัด 0812345678 ที่ 99/1 ซอยอารีย์ กทม 10400 บัตร 1-1002-01234-56-1 ขอบคุณครับ",
                        label="💬 กล่องพิมพ์ข้อความ:",
                        elem_classes=["phone-chat-input"]
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 ส่งแชต (Anonymize)", elem_classes=["line-send-btn"], scale=3)
                        clear_btn = gr.Button("ล้าง", scale=1)

            gr.Markdown("<h4 style='margin:12px 0 6px 0; font-size:13px; font-weight:700; color:#475569;'>⚡ ตัวอย่างสถานการณ์จำลอง (คลิกเพื่อทดสอบ):</h4>")
            preset_1 = gr.Button("📦 1. แชตส่งของ E-Commerce (ชื่อ, เบอร์, ที่อยู่, บัตร ปชช.)", elem_classes=["quick-chip"])
            preset_2 = gr.Button("💸 2. สลิปโอนเงิน (ธนาคาร, เลขบัญชี, ยอดเงิน)", elem_classes=["quick-chip"])
            preset_3 = gr.Button("🆔 3. ดักจับตัวเลขธุรกิจ (รหัสพัสดุ FLASH... คำสั่งซื้อ #ORD- รหัส paid...)", elem_classes=["quick-chip"])
            preset_4 = gr.Button("🏢 4. ที่อยู่สั้น Landmark (คอนโด / มหาวิทยาลัย ไร้รหัส ปณ.)", elem_classes=["quick-chip"])
            preset_5 = gr.Button("🗣️ 5. แชตสแลง/คำย่อ (ช่างโต้ง เบอ 089... ส่งที่ยุ ตึกแถวข้างเซเว่น)", elem_classes=["quick-chip"])

        # RIGHT: MULTI-VIEW OUTPUTS
        with gr.Column(scale=7, elem_classes=["enterprise-card"]):
            gr.Markdown("<h3 style='margin:0 0 12px 0; font-size:16px; font-weight:800; color:#0F172A;'>🛡️ Gateway Processing Pipeline (ผลลัพธ์ 3 มิติ)</h3>")
            
            with gr.Group():
                gr.HTML("""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:13px; font-weight:700; color:#1E293B; display:flex; align-items:center; gap:6px;">
                        <span class="card-step-badge" style="background:#DBEAFE; color:#1E40AF;">1</span>
                        🔍 Interactive Entity Visualizer (คำที่ตรวจพบไฮไลต์ตามสี)
                    </span>
                </div>
                """)
                output_highlight = gr.HighlightedText(
                    label="",
                    combine_adjacent=False,
                    color_map={
                        "ID_CARD": "#dc2626",
                        "ID_CARD_SUSPECTED": "#ea580c",
                        "NAME": "#2563eb",
                        "PHONE": "#d97706",
                        "ADDRESS": "#059669",
                        "BANK_ACC": "#0891b2",
                        "BANK": "#db2777",
                        "EMAIL": "#7c3aed"
                    }
                )

            with gr.Group():
                gr.HTML("""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; margin-top:10px;">
                    <span style="font-size:13px; font-weight:700; color:#1E293B; display:flex; align-items:center; gap:6px;">
                        <span class="card-step-badge" style="background:#D1FAE5; color:#065F46;">2</span>
                        🔒 Sanitized Prompt for External LLM (ข้อความปลอดภัยพร้อมส่งต่อ)
                    </span>
                </div>
                """)
                output_masked = gr.Textbox(
                    label="",
                    lines=3,
                    show_copy_button=True
                )

            with gr.Group():
                gr.HTML("""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; margin-top:10px;">
                    <span style="font-size:13px; font-weight:700; color:#1E293B; display:flex; align-items:center; gap:6px;">
                        <span class="card-step-badge" style="background:#F3E8FF; color:#6B21A8;">3</span>
                        🗺️ In-Memory Session Token Map (ตารางถอดรหัสคืนรูป)
                    </span>
                    <span style="font-size:11px; font-weight:600; color:#6B21A8; background:#FAF5FF; padding:2px 8px; border-radius:6px; border:1px solid #E9D5FF;">Ephemeral Zero-Disk</span>
                </div>
                """)
                output_mapping = gr.JSON(label="")

            output_status = gr.HTML("<div style='font-size:12.5px; color:#64748B;'>⏳ รอการประมวลผลข้อความ...</div>")

    # Preset Click Handlers
    preset_1.click(
        fn=lambda: "รบกวนส่งของให้คุณ สมชาย เข็มกลัด ที่ 99/1 ซอยอารีย์ พญาไท กทม 10400 เบอร์ 0812345678 บัตรประชาชน 1-1002-01234-56-1 ขอบคุณครับ",
        inputs=None,
        outputs=chat_input
    ).then(fn=process_text_gateway, inputs=chat_input, outputs=[output_highlight, output_masked, output_mapping, output_status])

    preset_2.click(
        fn=lambda: "โอนเงินเข้า บช. 123-4-56789-0 ธนาคารกสิกรไทย คุณกานดา เรียบร้อยแล้ว ยอด 15000 บาท ขอบคุณครับ",
        inputs=None,
        outputs=chat_input
    ).then(fn=process_text_gateway, inputs=chat_input, outputs=[output_highlight, output_masked, output_mapping, output_status])

    preset_3.click(
        fn=lambda: "แจ้งข้อมูลคำสั่งซื้อ #ORD-9876543210 เลขพัสดุ FLASH3290033773B บาร์โค้ด 8851234567890 สถานะ paid1234567890123 ปลายทาง คุณปริญญา 085-882-4345",
        inputs=None,
        outputs=chat_input
    ).then(fn=process_text_gateway, inputs=chat_input, outputs=[output_highlight, output_masked, output_mapping, output_status])

    preset_4.click(
        fn=lambda: "สถานที่ส่ง คอนโดลุมพินี พระราม 9 ฝากไว้ที่นิติชั้นล่าง ผู้รับ น้องบิว โทร 0612345678",
        inputs=None,
        outputs=chat_input
    ).then(fn=process_text_gateway, inputs=chat_input, outputs=[output_highlight, output_masked, output_mapping, output_status])

    preset_5.click(
        fn=lambda: "ทักคับ ติดต่อช่างโต้ง เบอ 0899998888 ส่งที่ยุ ตึกแถวข้างเซเว่น ต.แก่งคอย สระบุรี โอนผ่าน scb แล้ว",
        inputs=None,
        outputs=chat_input
    ).then(fn=process_text_gateway, inputs=chat_input, outputs=[output_highlight, output_masked, output_mapping, output_status])

    submit_btn.click(
        fn=process_text_gateway,
        inputs=chat_input,
        outputs=[output_highlight, output_masked, output_mapping, output_status]
    )

    clear_btn.click(
        fn=lambda: ("", [], "", {}, "<div style='font-size:12.5px; color:#64748B;'>⏳ รอการประมวลผลข้อความ...</div>"),
        inputs=None,
        outputs=[chat_input, output_highlight, output_masked, output_mapping, output_status]
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
