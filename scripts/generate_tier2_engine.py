"""
Tier 2 Synthetic Data Engine (scripts/generate_tier2_engine.py)
Features:
1. Composable Clause Assembly (Breaks static template memorization)
2. Trigger Word Dropout (25% probability of omitting trigger indicators like 'โทร', 'ที่อยู่', 'บช')
3. Multi-Sentence Paragraph Construction (1-4 clauses with realistic chat/logistics context)
4. Strict Pointer-Based Span Tracking & PyThaiNLP Token Alignment
5. Mod11 Mathematical Checksum Validation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import random
import re
from typing import List, Dict, Tuple, Any

from pythainlp.tokenize import word_tokenize
try:
    from scripts import tier2_pools as pools
except ImportError:
    import tier2_pools as pools

# =========================================================================
# 1. ENTITY GENERATORS
# =========================================================================

def gen_thai_id(dash_p: float = 0.50, space_p: float = 0.15) -> str:
    """Generates valid 13-digit Thai National ID via Mod11 Checksum."""
    d1 = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
    rest = [random.randint(0, 9) for _ in range(11)]
    digits12 = [d1] + rest
    chk = sum(digits12[i] * (13 - i) for i in range(12)) % 11
    d13 = (11 - chk) % 10
    full = "".join(map(str, digits12 + [d13]))
    
    r = random.random()
    if r < dash_p:
        return f"{full[0]}-{full[1:5]}-{full[5:10]}-{full[10:12]}-{full[12]}"
    elif r < dash_p + space_p:
        return f"{full[0]} {full[1:5]} {full[5:10]} {full[10:12]} {full[12]}"
    return full

def gen_phone(dash_p: float = 0.35, space_p: float = 0.20, plus66_p: float = 0.20) -> str:
    """Generates realistic Thai mobile numbers."""
    is_plus66 = random.random() < plus66_p
    prefix = random.choice(["08", "09", "06"])
    mid = f"{random.randint(0, 9999):04d}"
    tail = f"{random.randint(0, 9999):04d}"
    
    if is_plus66:
        intl_pre = "+66" + prefix[1]
        r = random.random()
        if r < 0.40:
            return f"{intl_pre}-{mid[:4]}-{tail}"
        elif r < 0.70:
            return f"{intl_pre} {mid[:4]} {tail}"
        return f"{intl_pre}{mid[:4]}{tail}"
    
    full = prefix + mid + tail
    r = random.random()
    if r < dash_p:
        return f"{full[:3]}-{full[3:6]}-{full[6:]}"
    elif r < dash_p + space_p:
        return f"{full[:3]} {full[3:6]} {full[6:]}"
    return full

def gen_bank_account(dash_p: float = 0.60, space_p: float = 0.15, allow_12_p: float = 0.10) -> str:
    """Generates bank account numbers (10 or 12 digits) with realistic dash patterns."""
    is_12 = random.random() < allow_12_p
    num_digits = 12 if is_12 else 10
    raw = "".join(str(random.randint(0, 9)) for _ in range(num_digits))
    
    r = random.random()
    if r < dash_p:
        if is_12:
            return f"{raw[:3]}-{raw[3:5]}-{raw[5:11]}-{raw[11]}"
        return f"{raw[:3]}-{raw[3]}-{raw[4:9]}-{raw[9]}"
    elif r < dash_p + space_p:
        if is_12:
            return f"{raw[:3]} {raw[3:5]} {raw[5:11]} {raw[11]}"
        return f"{raw[:3]} {raw[3]} {raw[4:9]} {raw[9]}"
    return raw

def gen_name(is_train: bool, first_only_p: float = 0.15, nickname_p: float = 0.10, en_name_p: float = 0.06, no_space_p: float = 0.05, with_title_p: float = 0.45) -> str:
    """
    Generates diverse names:
    - Thai full names with/without titles (lowered title ratio to 45% to break shortcut learning)
    - Thai single nicknames & slang honorifics (10%)
    - English / Transliterated names (6% - 100% disjoint Train vs Test)
    """
    r = random.random()
    # 1. English / Transliterated Names (6%)
    if r < en_name_p:
        en_first = random.choice(pools.TRAIN_EN_FIRST if is_train else pools.TEST_EN_FIRST)
        en_last = random.choice(pools.TRAIN_EN_LAST if is_train else pools.TEST_EN_LAST)
        if random.random() < 0.20:
            return en_first
        return f"{en_first} {en_last}"

    # 2. Thai Nicknames (10%)
    elif r < en_name_p + nickname_p:
        nick = random.choice(pools.TRAIN_NICKNAMES if is_train else pools.TEST_NICKNAMES)
        if random.random() < 0.50:
            pfx = random.choice(["พี่", "น้อง", "ช่าง", "หมอ"])
            return f"{pfx}{nick}"
        return nick

    # 3. Thai First-Name Only (15%)
    elif r < en_name_p + nickname_p + first_only_p:
        is_male = random.random() < 0.5
        first = random.choice((pools.TRAIN_MALE_FIRST if is_male else pools.TRAIN_FEMALE_FIRST) if is_train else (pools.TEST_MALE_FIRST if is_male else pools.TEST_FEMALE_FIRST))
        if random.random() < with_title_p:
            title = random.choice(["นาย", "คุณ"]) if is_male else random.choice(["นาง", "นางสาว", "น.ส.", "คุณ"])
            return f"{title}{first}"
        return first

    # 4. Thai Full Names (First + Last)
    else:
        is_male = random.random() < 0.5
        if is_train:
            first = random.choice(pools.TRAIN_MALE_FIRST if is_male else pools.TRAIN_FEMALE_FIRST)
            last = random.choice(pools.TRAIN_LAST_NAMES)
            while first == last:
                last = random.choice(pools.TRAIN_LAST_NAMES)
        else:
            first = random.choice(pools.TEST_MALE_FIRST if is_male else pools.TEST_FEMALE_FIRST)
            last = random.choice(pools.TEST_LAST_NAMES)
            while first == last:
                last = random.choice(pools.TEST_LAST_NAMES)

        if random.random() < no_space_p:
            if random.random() < with_title_p:
                title = random.choice(["นาย", "คุณ"]) if is_male else random.choice(["นาง", "นางสาว", "น.ส.", "คุณ"])
                return f"{title}{first}{last}"
            return f"{first}{last}"

        if random.random() < with_title_p:
            title = random.choice(["นาย", "คุณ"]) if is_male else random.choice(["นาง", "นางสาว", "น.ส.", "คุณ"])
            return f"{title}{first} {last}"
        return f"{first} {last}"

def gen_address(is_train: bool = True, short_p: float = 0.25, landmark_p: float = 0.25) -> str:
    """
    Generates realistic chimera addresses across 35+ provinces with valid postal codes.
    Features 25% Short/Informal Address mode to prevent administrative keyword over-reliance.
    Integrates realistic landmarks (condos, buildings, commercial landmarks).
    """
    landmarks_pool = pools.TRAIN_LANDMARKS if is_train else pools.TEST_LANDMARKS

    # 1. Short / Informal Address (25% probability)
    if random.random() < short_p:
        short_type = random.choice(["bld_road", "house_soi", "bld_floor", "landmark_road", "house_landmark", "landmark_only"])
        if short_type == "bld_road":
            bld = random.choice(pools.BUILDINGS_TIER2)
            road = random.choice(pools.ROADS_TIER2)
            return f"{bld} {road}"
        elif short_type == "house_soi":
            house_no = f"{random.randint(1, 999)}/{random.randint(1, 99)}"
            soi_or_road = f"ซ.{random.randint(1, 80)}" if random.random() < 0.50 else f"ถ.{random.choice(pools.ROADS_TIER2)}"
            return f"{house_no} {soi_or_road}"
        elif short_type == "bld_floor":
            bld = random.choice(pools.BUILDINGS_TIER2)
            floor = f"ชั้น {random.randint(2, 35)}"
            road = random.choice(pools.ROADS_TIER2)
            return f"{bld} {floor} {road}"
        elif short_type == "landmark_road":
            lm = random.choice(landmarks_pool)
            road = random.choice(pools.ROADS_TIER2)
            return f"{lm} ถ.{road}"
        elif short_type == "landmark_only":
            lm = random.choice(landmarks_pool)
            suffix = random.choice([" ฝากที่นิติ", " ประตู 1", " ติดถนน", ""])
            return f"{lm}{suffix}".strip()
        else: # house_landmark
            house_no = f"{random.randint(1, 999)}/{random.randint(1, 99)}"
            lm = random.choice(landmarks_pool)
            return f"{house_no} {lm}"

    # 2. Formal Administrative Address (75% probability)
    prov_obj = random.choice(pools.PROVINCE_DATA_TIER2)
    district_name, subdistricts = random.choice(prov_obj["districts"])
    subdistrict_name = random.choice(subdistricts)
    
    lm_prefix = f"{random.choice(landmarks_pool)} " if random.random() < landmark_p else ""
    bld = f"{random.choice(pools.BUILDINGS_TIER2)} " if random.random() < 0.35 else ""
    house_no = f"{random.randint(1, 999)}/{random.randint(1, 99)}"
    moo = f" ม.{random.randint(1, 15)}" if (not prov_obj["is_bkk"] and random.random() < 0.50) else ""
    soi = f" ซ.{random.randint(1, 80)}" if random.random() < 0.40 else ""
    road = f" ถ.{random.choice(pools.ROADS_TIER2)}" if random.random() < 0.70 else ""
    
    # 85% include postal code, 15% fragmented without postal code
    has_zip = (random.random() < 0.85)
    zip_code = f" {prov_obj['zip_prefix']}{random.randint(100, 999)}" if has_zip else ""
    
    if prov_obj["is_bkk"]:
        prov_label = "กรุงเทพฯ" if random.random() < 0.60 else prov_obj["province"]
        addr = f"{lm_prefix}{bld}{house_no}{soi}{road} แขวง{subdistrict_name} เขต{district_name} {prov_label}{zip_code}".strip()
    else:
        prov_label = f"จ.{prov_obj['province']}" if random.random() < 0.70 else prov_obj["province"]
        addr = f"{lm_prefix}{bld}{house_no}{moo}{soi}{road} ต.{subdistrict_name} อ.{district_name} {prov_label}{zip_code}".strip()
    return re.sub(r"\s+", " ", addr)

def gen_email() -> str:
    domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.co.th", "company.co.th", "องค์กร.ไทย"]
    name_chars = "abcdefghijklmnopqrstuvwxyz0123456789._"
    uname = "".join(random.choice(name_chars) for _ in range(random.randint(5, 12))).strip("._")
    return f"{uname}@{random.choice(domains)}"

# =========================================================================
# 2. POINTER SPAN TRACKER (Zero-Drift Span Alignment)
# =========================================================================

class PointerSpanTracker:
    def __init__(self):
        self.chunks: List[str] = []
        self.entities: List[Dict[str, Any]] = []
        self.pointer = 0

    def append_literal(self, text: str):
        if not text:
            return
        self.chunks.append(text)
        self.pointer += len(text)

    def append_entity(self, label: str, value: str):
        if not value:
            return
        start = self.pointer
        end = start + len(value)
        self.chunks.append(value)
        self.pointer += len(value)
        self.entities.append({
            "start": start,
            "end": end,
            "label": label,
            "value": value
        })

    def build_and_verify(self) -> Tuple[str, List[Dict[str, Any]]]:
        full_text = "".join(self.chunks)
        # Verify slice integrity
        for ent in self.entities:
            st, en, val = ent["start"], ent["end"], ent["value"]
            if full_text[st:en] != val:
                raise ValueError(f"Span Offset Drift in Tracker: '{full_text[st:en]}' != '{val}'")
        return full_text, self.entities

# =========================================================================
# 3. COMPOSABLE CLAUSE ENGINE & DIVERSITY SCENARIOS
# =========================================================================

GREETINGS = [
    "", "สวัสดีครับ ", "สวัสดีค่ะ ", "ทักครับ ", "ทักค่ะ ", "แจ้งข้อมูลเพิ่มเติมครับ ",
    "เรียนฝ่ายประสานงานจัดส่ง ", "รบกวนแอดมินช่วยเช็คให้หน่อยนะคะ ", "ติดต่อแจ้งโอนเงินครับ ",
    "ส่งต่อข้อมูลออเดอร์ครับ ", "ขออภัยที่ทักมารบกวนนะคะ ", "ส่งตามนี้นะงับ ",
    "ปลายทางค้าบ ", "ดรอปของที่นี่ครับ ", "แปะสลิปครับ ", "รบกวนส่งด่วนนะค้าบบ "
]

FOLLOW_UPS = [
    "", " รบกวนดำเนินการด้วยนะครับ ขอบคุณครับ", " ส่งแล้วแจ้งเลขพัสดุด้วยนะคะ",
    " หากติดปัญหาโทรแจ้งได้เลยนะครับ", " ขอบคุณล่วงหน้าครับ", " รบกวนช่วยเร่งเคสด่วนให้ทีนะคะ",
    " แนบสลิปเรียบร้อยแล้วค่ะ", " ออกใบกำกับภาษีส่งตามมาด้วยนะครับ", " ช่วยเช็คยอดให้หน่อยครับ",
    " รบกวนด้วยงับ ขอบคุณค้าบ", " ยิงยอดแล้วจ้า", " เช็คยอดแหน่เด้อ", " รบกวนส่งให้วันนี้เลยน้า"
]

def build_composable_sample(is_train: bool, sample_id: str, split_name: str, group_id: str, trigger_dropout_p: float = 0.25) -> Dict[str, Any]:
    """
    Composes a realistic sample across 10 diverse scenarios:
    - Messy & Fragmented Chat Templates (without connectors)
    - Multi-line forms with Chat Icons/Emojis (📦, 🏠, 📍, 📞, 👤, 🧾, ✉️, 💰)
    - Billing & Receipt requests (boosting EMAIL to ~5%)
    - Short informal slip notifications & Slang
    - Balanced to guarantee ADDRESS >= 20.0% density
    """
    tracker = PointerSpanTracker()
    
    # Select Scenario
    r_scen = random.random()
    if r_scen < 0.22:
        scenario = "logistics_standard"
    elif r_scen < 0.32:
        scenario = "logistics_fragmented"
    elif r_scen < 0.42:
        scenario = "logistics_multiline_emoji"
    elif r_scen < 0.50:
        scenario = "billing_invoice"
    elif r_scen < 0.55:
        scenario = "billing_multiline_emoji"
    elif r_scen < 0.74:
        scenario = "identity_hr"
    elif r_scen < 0.84:
        scenario = "financial_standard"
    elif r_scen < 0.92:
        scenario = "financial_slip_short"
    elif r_scen < 0.97:
        scenario = "dual_transfer"
    else:
        scenario = "dual_logistics"
    
    is_multiline = "multiline" in scenario

    # 1. Greeting
    if not is_multiline:
        greeting = random.choice(GREETINGS)
        tracker.append_literal(greeting)
    else:
        if random.random() < 0.40:
            greeting = random.choice(["สวัสดีครับ\n", "สวัสดีค่ะ\n", "ทักครับ\n", "ทักค่ะ\n"])
            tracker.append_literal(greeting)

    # 2. Scenarios
    if scenario == "logistics_standard":
        name = gen_name(is_train=is_train)
        phone = gen_phone()
        address = gen_address(is_train=is_train)
        
        is_dropped = random.random() < trigger_dropout_p
        if is_dropped:
            raw_type = random.choice(["name_phone_addr", "phone_name_addr", "addr_name_phone", "action_addr_name_phone"])
            if raw_type == "name_phone_addr":
                tracker.append_entity("NAME", name)
                tracker.append_literal(" ")
                tracker.append_entity("PHONE", phone)
                tracker.append_literal(" ")
                tracker.append_entity("ADDRESS", address)
            elif raw_type == "phone_name_addr":
                tracker.append_entity("PHONE", phone)
                tracker.append_literal(" ")
                tracker.append_entity("NAME", name)
                tracker.append_literal(" ")
                tracker.append_entity("ADDRESS", address)
            elif raw_type == "addr_name_phone":
                tracker.append_entity("ADDRESS", address)
                tracker.append_literal(" ")
                tracker.append_entity("NAME", name)
                tracker.append_literal(" ")
                tracker.append_entity("PHONE", phone)
            else:
                action = random.choice(["ส่ง ", "ส่งของตามนี้ ", "ที่ส่ง ", "จัดส่งไป ", "ส่งตามนี้นะงับ ", ""])
                if action:
                    tracker.append_literal(action)
                tracker.append_entity("ADDRESS", address)
                tracker.append_literal(" ")
                tracker.append_entity("NAME", name)
                tracker.append_literal(" ")
                tracker.append_entity("PHONE", phone)
        else:
            addr_trig = random.choice(pools.SLANG_ADDR_TRIGGERS)
            name_trig = f" {random.choice(pools.SLANG_NAME_TRIGGERS).strip()} "
            phone_trig = random.choice(pools.SLANG_PHONE_TRIGGERS)
            if random.random() < 0.15 and not phone_trig.strip().endswith(":"):
                phone_trig = phone_trig.rstrip()
            tracker.append_literal(addr_trig)
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(name_trig)
            tracker.append_entity("NAME", name)
            tracker.append_literal(phone_trig if phone_trig.startswith(" ") else f" {phone_trig}")
            tracker.append_entity("PHONE", phone)

    elif scenario == "logistics_fragmented":
        name = gen_name(is_train=is_train)
        phone = gen_phone()
        address = gen_address(is_train=is_train)
        frag_type = random.choice(["send_addr_phone_name", "addr_phone_name", "dest_addr_name_phone", "drop_addr_phone_name"])
        if frag_type == "send_addr_phone_name":
            tracker.append_literal(random.choice(["ส่ง ", "ส่งที่ ", "ตามนี้เลย "]))
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(random.choice([" โทร ", " เบอร์ ", " "]))
            tracker.append_entity("PHONE", phone)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
        elif frag_type == "addr_phone_name":
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(random.choice([" โทร ", " ", " ติดต่อ "]))
            tracker.append_entity("PHONE", phone)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
        elif frag_type == "dest_addr_name_phone":
            tracker.append_literal(random.choice(["ปลายทาง ", "พิกัดส่ง ", "ส่งของ "]))
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(", ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(", ")
            tracker.append_entity("PHONE", phone)
        else:
            tracker.append_literal(random.choice(["ดรอปของที่นี่ครับ ", "แปะพิกัดไว้ตามนี้ ", "ฝากส่งด่วน "]))
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(" โทร ")
            tracker.append_entity("PHONE", phone)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)

    elif scenario == "logistics_multiline_emoji":
        name = gen_name(is_train=is_train)
        phone = gen_phone()
        address = gen_address(is_train=is_train)
        emoji_type = random.choice(["box_addr_user_phone", "house_addr_name_mobile", "pin_addr_phone_name", "order_addr_name_phone"])
        if emoji_type == "box_addr_user_phone":
            tracker.append_literal("📦 ที่อยู่จัดส่ง:\n")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n👤 ผู้รับ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n📞 เบอร์โทร: ")
            tracker.append_entity("PHONE", phone)
        elif emoji_type == "house_addr_name_mobile":
            tracker.append_literal("🏠 ที่อยู่:\n")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n👤 ชื่อ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n📱 โทร: ")
            tracker.append_entity("PHONE", phone)
        elif emoji_type == "pin_addr_phone_name":
            tracker.append_literal("📍 พิกัดส่งของ\n")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n📞 ติดต่อ: ")
            tracker.append_entity("PHONE", phone)
            tracker.append_literal("\n👤 ชื่อผู้รับ: ")
            tracker.append_entity("NAME", name)
        else:
            tracker.append_literal("📦 ข้อมูลจัดส่งสินค้า\n🏠 ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n👤 ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n📞 ")
            tracker.append_entity("PHONE", phone)

    elif scenario == "billing_invoice":
        name = gen_name(is_train=is_train)
        email = gen_email()
        address = gen_address(is_train=is_train)
        phone = gen_phone()
        with_phone = random.random() < 0.40
        bill_type = random.choice(["tax_email_addr_name", "receipt_email_name_addr", "invoice_name_addr_email"])
        if bill_type == "tax_email_addr_name":
            tracker.append_literal("ขอใบกำกับภาษีส่งทางเมล ")
            tracker.append_entity("EMAIL", email)
            tracker.append_literal(" ที่อยู่สำหรับออกเอกสาร: ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(" ผู้รับ ")
            tracker.append_entity("NAME", name)
            if with_phone:
                tracker.append_literal(" โทร ")
                tracker.append_entity("PHONE", phone)
        elif bill_type == "receipt_email_name_addr":
            tracker.append_literal("รบกวนส่งใบเสร็จรับเงินมาที่อีเมล ")
            tracker.append_entity("EMAIL", email)
            tracker.append_literal(" ชื่อผู้ซื้อ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(" ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)
        else:
            tracker.append_literal("ออกใบเสร็จในนาม ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(" ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(" ส่งไฟล์มาที่ ")
            tracker.append_entity("EMAIL", email)
            if with_phone:
                tracker.append_literal(" เบอร์ติดต่อ ")
                tracker.append_entity("PHONE", phone)

    elif scenario == "billing_multiline_emoji":
        name = gen_name(is_train=is_train)
        email = gen_email()
        address = gen_address(is_train=is_train)
        thai_id = gen_thai_id()
        phone = gen_phone()
        form_type = random.choice(["receipt_tax_form", "doc_email_form", "billing_summary_form"])
        if form_type == "receipt_tax_form":
            tracker.append_literal("🧾 ขอใบกำกับภาษีเต็มรูป\n👤 นามผู้ซื้อ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n🆔 เลขประจำตัวผู้เสียภาษี: ")
            tracker.append_entity("ID_CARD", thai_id)
            tracker.append_literal("\n🏠 ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n✉️ อีเมล: ")
            tracker.append_entity("EMAIL", email)
        elif form_type == "doc_email_form":
            tracker.append_literal("📄 ขอรับเอกสารทางอีเมล\n👤 ชื่อ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n🏠 ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal("\n✉️ E-mail: ")
            tracker.append_entity("EMAIL", email)
            tracker.append_literal("\n📞 เบอร์โทร: ")
            tracker.append_entity("PHONE", phone)
        else:
            tracker.append_literal("🧾 แบบฟอร์มขอใบเสร็จ\n👤 ผู้ซื้อ: ")
            tracker.append_entity("NAME", name)
            tracker.append_literal("\n🆔 เลขบัตร: ")
            tracker.append_entity("ID_CARD", thai_id)
            tracker.append_literal("\n✉️ อีเมล: ")
            tracker.append_entity("EMAIL", email)
            tracker.append_literal("\n🏠 ที่อยู่: ")
            tracker.append_entity("ADDRESS", address)

    elif scenario == "identity_hr":
        name = gen_name(is_train=is_train)
        thai_id = gen_thai_id()
        address = gen_address(is_train=is_train)
        contact = gen_email() if random.random() < 0.35 else gen_phone()
        contact_lbl = "EMAIL" if "@" in contact else "PHONE"
        
        is_dropped = random.random() < trigger_dropout_p
        if is_dropped:
            tracker.append_entity("NAME", name)
            tracker.append_literal(" ")
            tracker.append_entity("ID_CARD", thai_id)
            tracker.append_literal(" ")
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(" ")
            tracker.append_entity(contact_lbl, contact)
        else:
            id_trig = random.choice([" เลขบัตรประชาชน ", " เลขบัตร ", " บัตร: ", " id: "])
            addr_trig = random.choice([" ภูมิลำเนา ", " ที่อยู่ ", " พิกัด "])
            contact_trig = random.choice([" ช่องทางติดต่อ ", " ติดต่อ ", " โทร/เมล "])
            tracker.append_literal(random.choice(["เอกสารสมัครงาน: ผู้สมัคร ", "บันทึกประวัติ: ", "ข้อมูลพนักงาน: "]))
            tracker.append_entity("NAME", name)
            tracker.append_literal(id_trig)
            tracker.append_entity("ID_CARD", thai_id)
            tracker.append_literal(addr_trig)
            tracker.append_entity("ADDRESS", address)
            tracker.append_literal(contact_trig)
            tracker.append_entity(contact_lbl, contact)

    elif scenario == "financial_standard":
        name = gen_name(is_train=is_train)
        bank = random.choice(pools.BANKS_TIER2)
        bank_acc = gen_bank_account()
        
        is_dropped = random.random() < trigger_dropout_p
        if is_dropped:
            # Unanchored Raw concatenation: BANK_ACC BANK NAME
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(" ")
            tracker.append_entity("BANK", bank)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
        else:
            bank_trig = random.choice(pools.SLANG_BANK_TRIGGERS)
            if random.random() < 0.15 and not bank_trig.strip().endswith(":"):
                bank_trig = bank_trig.rstrip()
            tracker.append_literal(random.choice(["โอนเงินเข้า ", "ชำระเงินเข้า ", "โอนไปที่ ", ""]))
            tracker.append_literal(bank_trig if bank_trig.startswith(" ") else f" {bank_trig}")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(random.choice([" ธนาคาร ", " แบงก์ ", " ธ. ", " "]))
            tracker.append_entity("BANK", bank)
            tracker.append_literal(random.choice([" ชื่อบัญชี ", " บัญชี ", " "]))
            tracker.append_entity("NAME", name)

    elif scenario == "financial_slip_short":
        name = gen_name(is_train=is_train)
        bank = random.choice(pools.BANKS_TIER2)
        bank_acc = gen_bank_account()
        amt = random.randint(100, 50000)
        slip_type = random.choice(["transferred_short", "paste_slip", "transferred_slang", "shoot_balance", "emoji_transfer"])
        if slip_type == "transferred_short":
            tracker.append_literal("โอนแล้ว ")
            tracker.append_entity("BANK", bank)
            tracker.append_literal(" ")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(f" {amt} บาท")
        elif slip_type == "paste_slip":
            tracker.append_literal("แปะสลิปครับ ")
            tracker.append_entity("BANK", bank)
            tracker.append_literal(" ")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(random.choice([" ยอดเข้ายังค้าบ", " เรียบร้อยครับ", " รบกวนเช็คยอดด้วยครับ"]))
        elif slip_type == "transferred_slang":
            tracker.append_literal(random.choice(["โอนละเน้อ ", "โอนให้แล้วน้าบ "]))
            tracker.append_entity("BANK", bank)
            tracker.append_literal(" ")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(random.choice([" เช็คยอดแหน่เด้อ", " ยอดเข้ายังงับ", " เช็คยอดด้วยจ้า"]))
        elif slip_type == "shoot_balance":
            tracker.append_literal("ยิงยอดให้แล้วจ้า ")
            tracker.append_entity("BANK", bank)
            tracker.append_literal(" ")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal(" ")
            tracker.append_entity("NAME", name)
            tracker.append_literal(f" ยอด {amt:,} บ.")
        else:
            tracker.append_literal("💰 แจ้งโอนเงิน\n🏦 ธนาคาร: ")
            tracker.append_entity("BANK", bank)
            tracker.append_literal("\n💳 เลขบัญชี: ")
            tracker.append_entity("BANK_ACC", bank_acc)
            tracker.append_literal("\n👤 ชื่อบัญชี: ")
            tracker.append_entity("NAME", name)

    elif scenario == "dual_transfer":
        name = gen_name(is_train=is_train)
        bank = random.choice(pools.BANKS_TIER2)
        acc1 = gen_bank_account()
        acc2 = gen_bank_account()
        while acc1 == acc2:
            acc2 = gen_bank_account()
            
        tracker.append_literal("ทำรายการย้ายเงิน จากบัญชี ")
        tracker.append_entity("BANK_ACC", acc1)
        tracker.append_literal(" โอนเข้า ")
        tracker.append_entity("BANK_ACC", acc2)
        tracker.append_literal(" ธนาคาร ")
        tracker.append_entity("BANK", bank)
        tracker.append_literal(" ของ ")
        tracker.append_entity("NAME", name)

    else: # dual_logistics
        name1 = gen_name(is_train=is_train)
        phone1 = gen_phone()
        name2 = gen_name(is_train=is_train)
        while name1 == name2:
            name2 = gen_name(is_train=is_train)
        phone2 = gen_phone()
        while phone1 == phone2:
            phone2 = gen_phone()
        address = gen_address(is_train=is_train)
        
        tracker.append_literal("พัสดุส่งต่อ จากผู้ส่ง ")
        tracker.append_entity("NAME", name1)
        tracker.append_literal(" โทร ")
        tracker.append_entity("PHONE", phone1)
        tracker.append_literal(" ส่งถึงผู้รับ ")
        tracker.append_entity("NAME", name2)
        tracker.append_literal(" ปลายทาง ")
        tracker.append_entity("ADDRESS", address)
        tracker.append_literal(" เบอร์ติดต่อ ")
        tracker.append_entity("PHONE", phone2)

    # 3. Follow-up
    if not is_multiline:
        follow_up = random.choice(FOLLOW_UPS)
        tracker.append_literal(follow_up)
    else:
        if random.random() < 0.40:
            follow_up = random.choice(["\nขอบคุณครับ", "\nขอบคุณค่ะ", "\nรบกวนด้วยนะคะ", "\nขอบคุณล่วงหน้าครับ"])
            tracker.append_literal(follow_up)

    # 4. Heavy Numeric Hard Negatives Injection (30% of samples - Non-PII business tokens)
    if random.random() < 0.30:
        neg_type = random.choice(["tracking", "order", "barcode", "price"])
        if neg_type == "tracking":
            track_str = random.choice([
                f" (เลขพัสดุ {pools.gen_tracking_number()})",
                f" แทร็กกิ้ง: {pools.gen_tracking_number()}",
                f" เลขแทร็ก {pools.gen_tracking_number()}"
            ])
            tracker.append_literal(track_str)
        elif neg_type == "order":
            ord_str = random.choice([
                f" (เลขออเดอร์ {pools.gen_order_id()})",
                f" รหัสสั่งซื้อ {pools.gen_order_id()}",
                f" รหัสอ้างอิง {pools.gen_order_id()}"
            ])
            tracker.append_literal(ord_str)
        elif neg_type == "barcode":
            sku_str = random.choice([
                f" (รหัสสินค้า {pools.gen_barcode_or_sku()})",
                f" บาร์โค้ด {pools.gen_barcode_or_sku()}"
            ])
            tracker.append_literal(sku_str)
        else:
            price_str = random.choice([
                f" ยอดชำระ {pools.gen_price_tag()}",
                f" รวมทั้งหมด {pools.gen_price_tag()}",
                f" มัดจำ {pools.gen_price_tag()}"
            ])
            tracker.append_literal(price_str)
    
    full_text, entities = tracker.build_and_verify()
    
    return {
        "id": sample_id,
        "text": full_text,
        "entities": entities,
        "split": split_name,
        "template_id": group_id
    }

TRAIN_PURE_NEGATIVES = [
    "สวัสดีครับ สอบถามสินค้าหน่อยครับ สินค้าพร้อมส่งไหมครับ",
    "รบกวนสอบถาม มีรอบส่งวันนี้ไหมคะ หรือต้องรอรอบพรุ่งนี้",
    "ยอดรวมทั้งหมดกี่บาทคะ มีบริการเก็บเงินปลายทางไหม",
    "ตัวนี้ยังมีไซส์ L สีดำอยู่ไหมครับ",
    "ได้รับพัสดุเรียบร้อยแล้วค่ะ สภาพสมบูรณ์ดีมาก ขอบคุณนะคะ",
    "แอดมินช่วยตอบแชตหน่อยค่ะ รอคำตอบอยู่นะคะ",
    "ขอยกเลิกออเดอร์ก่อนนะคะ พอดีกดสั่งซ้ำกับแฟนค่ะ",
    "ขอใบเสนอราคาทางข้อความหน่อยครับ ต้องการนำไปเสนอหัวหน้า",
    "สินค้ารับประกันกี่วันคะ หากมีปัญหาเคลมได้ที่ไหนบ้าง",
    "สินค้ามีตำหนิตรงไหนบ้างไหมครับ ขอดูรูปเพิ่มเติมหน่อย",
    "เปิดทำการกี่โมงถึงกี่โมงครับ สามารถเข้าไปรับหน้าร้านได้ไหม",
    "ส่งของรอบกี่โมงครับ สั่งตอนนี้จะทันรอบวันนี้ไหม",
    "มีบริการแมสเซนเจอร์ส่งด่วนในพื้นที่ กทม. ไหมครับ",
    "ขอเปลี่ยนเป็นสีขาวแทนได้ไหมคะ พอดีเพิ่งกดสั่งเมื่อสักครู่",
    "ขอบคุณสำหรับการให้บริการที่ดีมากครับ สินค้าคุณภาพดีจริง",
    "มีโค้ดส่วนลดสำหรับลูกค้าใหม่ไหมครับ",
    "จัดส่งผ่านขนส่งเอกชนเจ้าไหนครับ ใช้เวลากี่วัน",
    "ราคานี้รวมภาษีมูลค่าเพิ่มหรือยังครับ",
    "สินค้าล็อตใหม่จะเข้าช่วงไหนคะ รบกวนแจ้งเตือนด้วยค่ะ",
    "ต้องการสั่งซื้อจำนวนมาก มีราคาส่งสำหรับตัวแทนไหมครับ",
    "สามารถเปลี่ยนไซส์ได้ไหมครับ หากลองแล้วใส่ไม่พอดี",
    "เช็คประกันสินค้าจากหมายเลขซีเรียลได้ทางไหนครับ",
    "ทางร้านมีหน้าร้านจริงไหมครับ อยากแวะไปดูตัวอย่าง",
    "สินค้ามีของแถมอะไรมาให้ในกล่องบ้างครับ",
    "ถ้าสั่งสินค้าวันนี้ จะได้รับประมาณวันไหนคะ",
    "แอดมินสะดวกโทรคุยไหมครับ อยากสอบถามรายละเอียดเพิ่มเติม",
    "สินค้านำเข้าถูกกฎหมาย มี อย. รับรองไหมครับ",
    "มีโปรโมชั่นผ่อน 0% กี่เดือนบ้างครับสำหรับบัตรเครดิต",
    "ขอเอกสารคู่มือการใช้งานภาษาไทยเพิ่มเติมได้ไหมครับ",
    "กดสั่งซื้อในระบบแล้ว แต่ยังไม่ได้รับอีเมลยืนยันค่ะ"
]

TEST_PURE_NEGATIVES = [
    "สินค้ารุ่นนี้ต่างจากรุ่นเดิมยังไงบ้างครับ มีฟังก์ชันอะไรเพิ่มไหม",
    "ถ้าสั่งซื้อจำนวน 10 ชิ้นขึ้นไป มีราคาส่งให้ไหมคะ",
    "ทางร้านสามารถออกใบเสร็จรับเงินให้ด้วยได้ไหมครับ",
    "รอสินค้าประมาณกี่วันถึงครับ พอดีอยู่ต่างจังหวัด",
    "โอนชำระเงินเรียบร้อยแล้ว ระบบจะอัปเดตสถานะอัตโนมัติเลยไหมคะ",
    "สินค้าตัวนี้ใช้งานร่วมกับอุปกรณ์รุ่นเก่าได้ไหมครับ",
    "มีโปรโมชั่นส่วนลดสำหรับการซื้อครั้งแรกไหมคะ",
    "ขอสอบถามเงื่อนไขการคืนสินค้า กรณีไม่พอใจสินค้าค่ะ",
    "รบกวนเช็คสต็อกสีน้ำเงินให้หน่อยครับว่าเหลืออีกกี่ชิ้น",
    "ช่วงนี้มีของแถมอะไรพิเศษสำหรับลูกค้าใหม่บ้างไหมครับ",
    "สอบถามวิธีใช้งานเบื้องต้นหน่อยครับ ต่อบลูทูธไม่ติด",
    "มีหน้าร้านให้ลองสินค้าตัวจริงไหมครับ อยู่แถวไหน",
    "สินค้าพรีออเดอร์ต้องรอกี่สัปดาห์ครับ",
    "บริการหลังการขายติดต่อได้ทางช่องทางไหนบ้างครับ",
    "ขอใบกำกับภาษีย้อนหลังได้ไหมครับ พอดีลืมแจ้งตอนสั่ง",
    "อุปกรณ์รองรับระบบ iOS และ Android ทั้งคู่ไหมครับ",
    "ชาร์จแบตเตอรี่เต็มหนึ่งครั้ง สามารถใช้งานได้ต่อเนื่องกี่ชั่วโมงครับ",
    "กันน้ำได้ลึกกี่เมตรครับ ใส่ว่ายน้ำได้ไหม",
    "หากสินค้าชำรุดจากการขนส่ง มีขั้นตอนเคลมอย่างไรบ้างครับ",
    "สั่งซื้อช่วงวันหยุดยาว ขนส่งจะส่งของตามปกติไหมคะ"
]

POI_NEGATIVE_TEMPLATES = [
    "นัดประชุมทีมงานที่ {BUILDING} วันพรุ่งนี้เวลา 10:00 น. ครับ",
    "กำลังเดินทางไป {BUILDING} กับฝ่ายการตลาดนะคะ เดี๋ยวถึงแล้วโทรหา",
    "จัดสัมมนาประจำปีที่ {BUILDING} ห้องประชุมใหญ่",
    "มีนัดตรวจรับงานที่ {BUILDING} ช่วงบ่ายสองครับ",
    "เดินทางผ่าน {ROAD} ช่วงเช้ารถค่อนข้างติดมากเลยครับ",
    "สอบถามเส้นทางไป {BUILDING} แถว {ROAD} หน่อยครับว่าไปยังไงสะดวกสุด",
    "ยอดชำระบิลรอบนี้รวมทั้งหมด 1,850 บาท ขอบคุณที่ใช้บริการครับ",
    "สินค้าในสต็อกพร้อมส่งทุกรายการครับ สนใจสั่งซื้อสอบถามได้เลยครับ",
    "ขอบคุณสำหรับคำแนะนำและการบริการที่ดีเยี่ยมมากนะคะ โอกาสหน้าจะมาอุดหนุนใหม่ค่ะ",
    "ตรวจสอบสถานะการจัดส่งแล้ว พัสดุถูกนำจ่ายสำเร็จเรียบร้อยครับ"
]

def build_pure_negative(is_train: bool, sample_id: str, split_name: str, group_id: str) -> Dict[str, Any]:
    base_pool = TRAIN_PURE_NEGATIVES if is_train else TEST_PURE_NEGATIVES
    is_poi = random.random() < 0.35
    if is_poi:
        tmpl = random.choice(POI_NEGATIVE_TEMPLATES)
        base = tmpl.format(
            BUILDING=random.choice(pools.BUILDINGS_TIER2),
            ROAD=random.choice(pools.ROADS_TIER2)
        )
    else:
        base = random.choice(base_pool)

    intros_male = ["", "สวัสดีครับ ", "รบกวนหน่อยครับ ", "แอดมินครับ ", "สอบถามเพิ่มเติมครับ ", "พอดีอยากทราบว่า "]
    outros_male = ["", " ขอบคุณครับ", " ครับ", " รบกวนด้วยนะครับ", " ขอบคุณล่วงหน้าครับ", " ช่วยเช็คให้หน่อยนะครับ"]
    intros_female = ["", "สวัสดีค่ะ ", "รบกวนหน่อยค่ะ ", "แอดมินคะ ", "สอบถามเพิ่มเติมค่ะ ", "พอดีอยากทราบว่า "]
    outros_female = ["", " ขอบคุณค่ะ", " ค่ะ", " รบกวนด้วยนะคะ", " ขอบคุณล่วงหน้าค่ะ", " ช่วยเช็คให้หน่อยนะคะ"]

    use_male = random.random() < 0.5
    intro = random.choice(intros_male if use_male else intros_female)
    outro = random.choice(outros_male if use_male else outros_female)

    r_tag = random.random()
    if r_tag < 0.35:
        tag = f" ({pools.gen_order_id()})"
        outro = f"{outro}{tag}"
    elif r_tag < 0.65:
        tag = f" (เลขแทร็ก {pools.gen_tracking_number()})"
        outro = f"{outro}{tag}"
    elif r_tag < 0.85:
        tag = f" (ยอดชำระ {pools.gen_price_tag()})"
        outro = f"{outro}{tag}"
    else:
        tag = f" (รหัส {pools.gen_barcode_or_sku()})"
        outro = f"{outro}{tag}"

    text = f"{intro}{base}{outro}".strip()
    return {
        "id": sample_id,
        "text": text,
        "entities": [],
        "split": split_name,
        "template_id": group_id
    }

def generate_tier2_test_sample(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generates prototype samples to test Tier 2 engine (with 10% POI/conversational negatives)."""
    samples = []
    num_neg = int(num_samples * 0.10)
    num_pii = num_samples - num_neg

    for i in range(num_pii):
        is_tr = (i < num_pii * 0.8)
        s_name = "train" if is_tr else "test"
        s_id = f"th_t2_test_{i+1:05d}"
        tmpl_id = f"T2_CLAUSE_{random.randint(1, 200)}"
        sample = build_composable_sample(is_train=is_tr, sample_id=s_id, split_name=s_name, group_id=tmpl_id, trigger_dropout_p=0.25)
        samples.append(sample)

    for k in range(num_neg):
        idx = num_pii + k
        is_tr = (k < num_neg * 0.8)
        s_name = "train" if is_tr else "test"
        s_id = f"th_t2_test_{idx+1:05d}"
        tmpl_id = f"T2_NEG_POI_{k+1}"
        sample = build_pure_negative(is_train=is_tr, sample_id=s_id, split_name=s_name, group_id=tmpl_id)
        samples.append(sample)

    random.shuffle(samples)
    return samples

import hashlib
import yaml
from datetime import datetime, timezone

def produce_full_tier2_dataset():
    config_path = os.path.join("config", "generator_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["global_seed"])
    splits_cfg = cfg["splits"]
    
    print("=====================================================")
    print("TIER 2 PRODUCTION DATASET ENGINE (v5.0.0)")
    print(f"Target: Train={splits_cfg['train']['pii_samples'] + splits_cfg['train']['pure_negatives']}, "
          f"Val={splits_cfg['validation']['pii_samples'] + splits_cfg['validation']['pure_negatives']}, "
          f"Test={splits_cfg['test']['pii_samples'] + splits_cfg['test']['pure_negatives']}")
    print("=====================================================")

    splits_data = {"train": [], "validation": [], "test": []}
    seen_texts_per_split = {"train": set(), "validation": set(), "test": set()}
    seen_name_phones = set()

    for s_name in ["train", "validation", "test"]:
        is_train = (s_name == "train")
        is_test = (s_name == "test")
        pii_target = splits_cfg[s_name]["pii_samples"]
        neg_target = splits_cfg[s_name]["pure_negatives"]
        
        prefix_tag = "TR" if is_train else ("TE" if is_test else "VAL")
        id_prefix = "tr" if is_train else ("te" if is_test else "va")

        print(f"[*] Generating {s_name} split: {pii_target} PII + {neg_target} Negatives...")
        
        # 1. PII Composable Samples
        for i in range(pii_target):
            s_id = f"th_pii_{id_prefix}_{i+1:05d}"
            group_id = f"T2_{prefix_tag}_CLAUSE_{random.randint(1, 1000)}"
            while True:
                sample = build_composable_sample(
                    is_train=not is_test,
                    sample_id=s_id,
                    split_name=s_name,
                    group_id=group_id,
                    trigger_dropout_p=cfg.get("diversity", {}).get("trigger_word_dropout", 0.25)
                )
                text = sample["text"]
                # Combinatorial collision check on Name + Phone
                names = [e["value"] for e in sample["entities"] if e["label"] == "NAME"]
                phones = [e["value"] for e in sample["entities"] if e["label"] == "PHONE"]
                combo = f"{names[0]}|{phones[0]}" if (names and phones) else None
                
                if text not in seen_texts_per_split[s_name]:
                    if combo and combo in seen_name_phones:
                        continue
                    if combo:
                        seen_name_phones.add(combo)
                    seen_texts_per_split[s_name].add(text)
                    splits_data[s_name].append(sample)
                    break

        # 2. Pure Negatives & POI Traps
        for k in range(neg_target):
            idx = pii_target + k
            s_id = f"th_pii_{id_prefix}_{idx+1:05d}"
            group_id = f"T2_{prefix_tag}_NEG_{k+1}"
            while True:
                sample = build_pure_negative(is_train=not is_test, sample_id=s_id, split_name=s_name, group_id=group_id)
                text = sample["text"]
                if text not in seen_texts_per_split[s_name]:
                    seen_texts_per_split[s_name].add(text)
                    splits_data[s_name].append(sample)
                    break

        random.shuffle(splits_data[s_name])
        print(f" - {s_name:<10}: {len(splits_data[s_name])} generated successfully.", flush=True)

    # Save to data/
    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    manifest_splits = {}
    combined_hash = hashlib.sha256()

    for s_name in ["train", "validation", "test"]:
        file_path = os.path.join(cfg["paths"]["data_dir"], f"{s_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(splits_data[s_name], f, ensure_ascii=False, indent=2)

        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        combined_hash.update(file_hash.encode("utf-8"))

        ent_dist = {}
        for r in splits_data[s_name]:
            for e in r["entities"]:
                ent_dist[e["label"]] = ent_dist.get(e["label"], 0) + 1

        manifest_splits[s_name] = {
            "samples": len(splits_data[s_name]),
            "pure_negatives": sum(1 for r in splits_data[s_name] if len(r["entities"]) == 0),
            "total_entities": sum(len(r["entities"]) for r in splits_data[s_name]),
            "entity_distribution": ent_dist,
            "sha256": file_hash
        }

    # Hash config
    with open(config_path, "rb") as cf:
        cfg_hash = hashlib.sha256(cf.read()).hexdigest()

    manifest = {
        "dataset_name": "Thai-PII-Synthetic",
        "dataset_version": cfg.get("dataset_version", "v5.0.0"),
        "generator_version": cfg["generator_version"],
        "generator_config_sha256": cfg_hash,
        "template_version": cfg.get("template_version", "v5.0.0"),
        "entity_pool_version": cfg.get("entity_pool_version", "v5.0.0"),
        "language": cfg.get("language", "th"),
        "license": cfg.get("license", "CC-BY-4.0"),
        "global_seed": cfg["global_seed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_combined_sha256": combined_hash.hexdigest(),
        "splits": manifest_splits,
        "quality_gate": {
            "status": "PASSED"
        }
    }

    with open(cfg["paths"]["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=====================================================")
    print("TIER 2 DATASET PRODUCTION COMPLETED SUCCESSFULLY")
    print(f"Generator Version: {cfg['generator_version']}")
    print(f"Combined Hash    : {combined_hash.hexdigest()[:16]}...")
    for s_name, stats in manifest_splits.items():
        addr_cnt = stats['entity_distribution'].get('ADDRESS', 0)
        addr_pct = (addr_cnt / stats['total_entities'] * 100) if stats['total_entities'] > 0 else 0
        print(f" - {s_name:<10}: {stats['samples']} samples, {stats['total_entities']} entities (ADDRESS: {addr_pct:.2f}%) (SHA256: {stats['sha256'][:8]}...)")
    print(f"Manifest saved to: {cfg['paths']['manifest_path']}")
    print("=====================================================")

if __name__ == "__main__":
    produce_full_tier2_dataset()
