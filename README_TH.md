---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- token-classification
task_ids:
- named-entity-recognition
language:
- th
license: cc-by-4.0
tags:
- thai
- ner
- pii
- synthetic-data
- privacy
- privacy-preserving-synthetic-data
- token-classification
pretty_name: Thai PII Synthetic NER Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.json
  - split: validation
    path: data/validation.json
  - split: test
    path: data/test.json
- config_name: iob2
  data_files:
  - split: train
    path: data/hf_export/train.json
  - split: validation
    path: data/hf_export/validation.json
  - split: test
    path: data/hf_export/test.json
---

<div align="center">

# Thai-PII-Synthetic
### A Synthetic Dataset for Thai Named Entity Recognition and Privacy De-Identification

Thai-PII-Synthetic เป็นชุดข้อมูลสังเคราะห์จำนวน 20,000 ตัวอย่าง (60,677 entities) สำหรับงานการรู้จำหน่วยงานที่มีชื่อ (NER) และการจำแนกคำระดับโทเคนภาษาไทย เพื่อการตรวจจับข้อมูลส่วนบุคคล 7 ประเภท โดยสร้างขึ้นจากอัลกอริทึมจำลอง ปราศจากข้อมูลจริงของบุคคลธรรมดา รองรับทั้งรูปแบบ Character-level Spans และ Pre-tokenized IOB2

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic)
[![Task](https://img.shields.io/badge/Task-Token_Classification_%2F_NER-blue.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Privacy_%2F_Synthetic_Dataset-green.svg)]()
[![Version](https://img.shields.io/badge/Version-v6.0.0-blue.svg)]()
[![English Docs](https://img.shields.io/badge/Docs-English-blue.svg)](README.md)

<p align="center">
  <a href="#dataset-summary">Summary</a> •
  <a href="#example-record">Example</a> •
  <a href="#data-fields-and-splits">Fields & Splits</a> •
  <a href="#usage">Usage</a> •
  <a href="#baseline">Baseline</a> •
  <a href="#ข้อกำหนดการใช้งานและความปลอดภัย-responsible-use--safety">Safety & Policy</a> •
  <a href="#citation">Citation</a>
</p>

</div>

---

## Dataset Summary

ชุดข้อมูลสังเคราะห์ภาษาไทยจำนวน **20,000 ประโยค (60,677 entities)** เพื่อการประเมินและฝึกฝนระบบตรวจจับข้อมูลส่วนบุคคล (**Named Entity Recognition: NER**) และการปกปิดข้อมูลอัตโนมัติ (De-identification) ก่อนส่งข้อความเข้าสู่โมเดลภาษาขนาดใหญ่ (Pre-LLM Privacy Gateways) โดยอ้างอิงแนวคิดเทคโนโลยีเสริมสร้างความเป็นส่วนตัว (*Privacy Enhancing Technologies: PETs*) จากฉบับร่างแนวปฏิบัติของ สคส. ด้าน AI

| รายละเอียด (Attribute) | ข้อมูลจำเพาะ (Specification) |
| :--- | :--- |
| **Tasks** | Named Entity Recognition (NER), Token Classification, PII De-identification |
| **Primary Language** | ภาษาไทย (`th`) |
| **Total Samples** | **20,000 ประโยค** (Train: 16,000 / Validation: 2,000 / Test: 2,000) |
| **Total Entities** | **60,677 entities** (Train: 48,586 / Validation: 6,039 / Test: 6,052) |
| **Entity Labels (7)** | `NAME`, `ID_CARD`, `PHONE`, `EMAIL`, `BANK_ACC`, `BANK`, `ADDRESS` |
| **Diversity Upgrades (v6)** | 50% Trigger Word Dropout, Non-PII Hard Negatives (เลขพัสดุ, คำสั่งซื้อ, บาร์โค้ด), ชื่อเล่น/ชื่ออังกฤษ, ที่อยู่สั้นและ Landmark |
| **Data Formats** | 1. `default`: Character-level Span JSON<br>2. `iob2`: Pre-tokenized IOB2 JSON (15 classes) |
| **Disjoint Splitting** | Template & Lexical Overlap = 0% ข้าม Split (Train: 2,600 / Val: 1,037 / Test: 1,035 เทมเพลต) |
| **License** | [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) |
| **Maintainer** | `DuckerMaster` |

### มีอะไรใหม่ใน v6.0.0 (Adversarial Robustness Upgrades)
เพื่อแก้ปัญหา Shortcut Learning (การที่โมเดลจำคำบอกใบ้หน้า PII แทนที่จะเข้าใจโครงสร้างข้อความจริง) ใน v6.0.0 ได้เพิ่ม:
1. **50% Trigger Word Dropout:** สุ่มตัดคำบอกใบ้ (`โทร:`, `ชื่อ:`, `ที่อยู่:`, `เลขบัญชี:`) 50% ของตัวอย่าง เพื่อจำลองการตัดแปะข้อความแชตลอยๆ
2. **30% Numeric Hard Negatives:** ฉีดตัวเลขรหัสพัสดุ (เช่น `TH0192837465`), รหัสคำสั่งซื้อ (`#ORD-91823746`), บาร์โค้ดสินค้า (`SKU-481920`), และยอดเงิน เข้าไปเป็น Non-PII (`O`) ป้องกัน False Positive
3. **ความสมจริงของชื่อ:** เพิ่มชื่อเล่นไทย 10% (เช่น `พี่โต้ง`, `หมอเอก`), ชื่ออังกฤษ/ทับศัพท์ 6%, และลดคำนำหน้านามทางการเหลือ 45%
4. **ที่อยู่สั้นและ Landmark (25%):** รองรับที่อยู่แบบระบุชื่อตึก คอนโด ชุมชน หรือจุดสังเกต โดยไม่มีตำบล/อำเภอ
5. **คำสแลงแชตและการตัดช่องว่าง:** รองรับคำสะกดแชตจริง (`เบอ`, `บช`, `ที่ยุ`) และตัดช่องว่างเชื่อมติดกัน

---

## Example Record

ตัวอย่างข้อมูลจาก `default` configuration (Character-level Spans):

```json
{
  "id": "th_pii_tr_01676",
  "text": "แจ้งข้อมูลเพิ่มเติมครับ รบกวนส่งใบเสร็จรับเงินมาที่อีเมล o41kjnpr@yahoo.com ชื่อผู้ซื้อ: นเรศ เพชรประเสริฐ ที่อยู่: สยามทาวเวอร์ ชั้น 9 งามวงศ์วาน รบกวนดำเนินการด้วยนะครับ ขอบคุณครับ",
  "entities": [
    {
      "start": 57,
      "end": 75,
      "label": "EMAIL",
      "value": "o41kjnpr@yahoo.com"
    },
    {
      "start": 89,
      "end": 106,
      "label": "NAME",
      "value": "นเรศ เพชรประเสริฐ"
    },
    {
      "start": 116,
      "end": 146,
      "label": "ADDRESS",
      "value": "สยามทาวเวอร์ ชั้น 9 งามวงศ์วาน"
    }
  ],
  "split": "train",
  "template_id": "T2_TR_CLAUSE_777"
}
```

---

## Data Fields and Splits

### Data Fields
* `id` *(string)*: รหัสระบุตัวอย่างข้อมูล เช่น `th_pii_tr_00001`
* `text` *(string)*: ข้อความภาษาไทย (บทสนทนา, ฟอร์มสั่งซื้อ, สลิปโอนเงิน)
* `entities` *(list)*: รายการเอนทิตีที่ปรากฏในข้อความ ประกอบด้วย:
  * `start` *(int)*: ตำแหน่งอักขระเริ่มต้น (0-indexed)
  * `end` *(int)*: ตำแหน่งอักขระสิ้นสุด (Python slice convention)
  * `label` *(string)*: ประเภทเอนทิตี (1 ใน 7 ประเภท)
  * `value` *(string)*: ค่าของเอนทิตีที่ถูกตัดตรงกับ `text[start:end]`
* `split` *(string)*: กลุ่มข้อมูล (`train`, `validation`, `test`)
* `template_id` *(string)*: รหัสกลุ่มเทมเพลตที่ใช้สังเคราะห์
* *(เฉพาะ `iob2` config)* เพิ่ม `tokens` (ตัดคำด้วย `pythainlp.tokenize.newmm`), `ner_tags` (int IDs), และ `ner_labels` (IOB2 15 classes)

### Entity Distribution
บันทึกการตรวจสอบสถิติการกระจายตัวของเอนทิตีจาก `data/manifest.json`:

| Entity Type | Train (16k) | Validation (2k) | Held-out Test (2k) | Total Entities | สัดส่วน |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NAME** | 14,827 | 1,843 | 1,841 | 18,511 | 30.51% |
| **ADDRESS** | 11,130 | 1,382 | 1,390 | 13,902 | 22.91% |
| **PHONE** | 9,289 | 1,129 | 1,128 | 11,546 | 19.03% |
| **BANK_ACC** | 3,983 | 514 | 492 | 4,989 | 8.22% |
| **BANK** | 3,270 | 418 | 410 | 4,098 | 6.75% |
| **ID_CARD** | 3,282 | 397 | 420 | 4,099 | 6.76% |
| **EMAIL** | 2,805 | 356 | 371 | 3,532 | 5.82% |
| **Total Entities** | **48,586** | **6,039** | **6,052** | **60,677** | **100.00%** |

> **หมายเหตุความหนาแน่น (Density Balance):** สัดส่วนของ `ADDRESS` คิดเป็น **22.91%** (ผ่านเกณฑ์ควบคุมความหนาแน่น ≥ 20%) และสัดส่วน `EMAIL` คิดเป็น **5.82%** (3,532 / 60,677)

### Dataset Splits and Composition
แจกแจงจำนวนตัวอย่างประโยค, ตัวอย่างเชิงลบ (Negatives), และการแบ่งกลุ่มเทมเพลต:

| รายการ (Metric) | Train Split | Validation Split | Held-out Test Split | รวมทั้งหมด (Total) |
| :--- | :---: | :---: | :---: | :---: |
| **จำนวนประโยคทั้งหมด** | 16,000 | 2,000 | 2,000 | **20,000 ประโยค** |
| **ประโยคที่มีข้อมูล PII** | 14,400 | 1,800 | 1,800 | 18,000 (90.0%) |
| **Pure Negatives (0 PII)** | 1,600 (10.0%) | 200 (10.0%) | 200 (10.0%) | 2,000 (10.0%) |
| **Hard Numeric Negatives** | ~4,320 | ~540 | ~540 | ~5,400 (30% ของประโยค PII) |
| **Disjoint Group Templates** | 2,600 (`T2_TR_*`) | 1,037 (`T2_VAL_*`) | 1,035 (`T2_TE_*`) | 4,672 (0% Overlap) |

---

## Usage

### 1. โหลดข้อมูลแบบ Character Span (Default Format)

```python
from datasets import load_dataset

# โหลดชุดข้อมูลหลักแบบ Character Span
dataset = load_dataset("DuckerMaster/Thai-PII-Synthetic", "default")
print(dataset)
print("Sample:", dataset["train"][0])
```

### 2. โหลดข้อมูลแบบ Token Classification (IOB2 Format)

```python
from datasets import load_dataset

# โหลดชุดข้อมูลแบบ Pre-tokenized IOB2 สำหรับ Fine-tune Transformer
token_dataset = load_dataset("DuckerMaster/Thai-PII-Synthetic", "iob2")
print("Tokens :", token_dataset["train"][0]["tokens"][:8])
print("Labels :", token_dataset["train"][0]["ner_labels"][:8])
```

---

## Baseline

ผลการทดสอบระบบ **Regex Baseline** บน **Held-out Test Set (ชื่อและเทมเพลตไม่ซ้ำกับ Train: 2,000 ตัวอย่าง / 6,052 Entities)** จากรายงาน `reports/evaluation_report.json`:

| Entity Type | Exact Precision | Exact Recall | Exact F1 | Partial Overlap F1 | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NAME** | 0.7402 | 0.2863 | 0.4128 | **0.4740** | 1,841 |
| **ID_CARD** | 0.8049 | 0.8548 | **0.8291** | 0.8291 | 420 |
| **PHONE** | 0.9868 | 0.8626 | **0.9205** | 0.9205 | 1,128 |
| **EMAIL** | 1.0000 | 0.7978 | **0.8876** | 0.8876 | 371 |
| **BANK_ACC** | 0.6667 | 0.7683 | **0.7139** | 0.7139 | 492 |
| **BANK** | 0.8006 | 0.6854 | **0.7385** | 0.7385 | 410 |
| **ADDRESS** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 1,390 |
| **Overall (Micro Avg)** | **0.5981** | **0.5981** | **0.5981** | **0.6519** | **6,052** |

* **ข้อสังเกต:** Regex ไม่สามารถตรวจจับ `ADDRESS` ได้ (`F1 = 0.0000`) และ Recall ของ `NAME` ลดลงเหลือ 0.2863 จากการตัดคำนำหน้าออก 50% สะท้อนว่าจำเป็นต้องใช้ Contextualized Representation (เช่น WangchanBERTa) ในการเรียนรู้บริบท
* **ชุดทดสอบสัญญาณรบกวนเบื้องต้น:** เมื่อทดสอบกับชุดข้อความจำลองคำพิมพ์ผิดและช่องว่างเกิน Exact Micro F1 เปลี่ยนแปลงเล็กน้อยจาก 0.5981 เป็น 0.5975 (Delta = -0.0006)

---

## ข้อกำหนดการใช้งานและความปลอดภัย (Responsible Use & Safety)

* **ข้อความปฏิเสธความรับผิดชอบและข้อห้าม (Synthetic Disclaimer):** ข้อมูลทั้งหมดเป็นข้อมูลสังเคราะห์ 100% เพื่อการวิจัยด้าน Privacy NLP แม้ชื่อ-ที่อยู่จะสุ่มขึ้นมา แต่เลขบัตรและเบอร์โทรสร้างตามโครงสร้างจริงของไทย จึง**ห้ามนำไปติดต่อ โอนเงิน สมัครบริการ หลอกลวง (Fraud) หรือทำธุรกรรมจริงโดยเด็ดขาด**
* **ขอบเขตตาม PDPA:** ครอบคลุมตัวอย่าง PII ทั่วไป 7 หมวดที่พบบ่อย ไม่รวมข้อมูลชีวมิติหรือประวัติสุขภาพ องค์กรผู้ใช้งานต้องประเมินความเสี่ยง (DPIA) ของตนเอง
* **นโยบายความปลอดภัยฉบับเต็ม:** ดูรายละเอียดข้อห้ามการใช้งานและช่องทางแจ้งช่องโหว่เพิ่มเติมได้ที่ [SAFETY.md](SAFETY.md)

---

## Citation

### BibTeX
```bibtex
@misc{duckermaster2026thaipii,
  author       = {DuckerMaster},
  title        = {Thai PII Synthetic NER Dataset},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic}},
  note         = {Version 6.0.0}
}
```

### Regulatory and Academic References
1. **PDPC Thailand:** *แนวปฏิบัติการคุ้มครองข้อมูลส่วนบุคคลเกี่ยวกับการพัฒนาและการใช้งานเทคโนโลยีปัญญาประดิษฐ์ (AI)* (ฉบับร่างแนวปฏิบัติ), สำนักงานคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล (สคส.).
2. **NIST SP 800-122:** McCallister, E., Grance, T., & Scarfone, K. (2010). *Guide to Protecting the Confidentiality of Personally Identifiable Information (PII)*. National Institute of Standards and Technology.
3. **Article 29 Data Protection Working Party (WP216 / 2014):** *Opinion 05/2014 on Anonymisation Techniques*, European Data Protection Board (EDPB).
4. **PIIBench (2026):** *A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection*, arXiv:2604.15776.
5. **SPY Benchmark (2025):** Savkin, M., Ionov, T., & Konovalov, V. (2025). *SPY: Enhancing Privacy with Synthetic PII Detection Dataset*, In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop (NAACL 2025 SRW), pages 236–246. https://aclanthology.org/2025.naacl-srw.23/
