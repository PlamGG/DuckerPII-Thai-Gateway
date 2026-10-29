<div align="center">

# 🌪️ DuckerPII: High-Performance Thai PII De-Identification & Privacy Gateway

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/DuckerMaster/DuckerPII-Thai-Base)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space_Demo-purple)](https://huggingface.co/spaces/DuckerMaster/DuckerPII-Thai-Gateway)
[![CI Tests](https://img.shields.io/badge/CI%2FCD-41%2F41_Passing-brightgreen.svg)]()

<p align="center">
  <a href="#news">News</a> •
  <a href="#model-description">Description</a> •
  <a href="#benchmark-results">Benchmark</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#safety--limitations">Limitations</a> •
  <a href="#citation">Citation</a>
</p>

</div>

---

**DuckerPII** is an open-source, enterprise-grade Thai privacy de-identification and governance suite developed by **DuckerMaster**. Designed specifically as a low-latency **Pre-LLM Privacy Reverse Proxy**, it detects, redacts, and synthesizes 7 foundational categories of Personally Identifiable Information (**PII**) in unstructured Thai text, ensuring strict compliance with Thailand's **Personal Data Protection Act (PDPA)** and draft PDPC AI regulatory guidelines.

---

## 📢 News

* **[October 2026]** Released **DuckerPII-Thai-Base** model weights and **Thai-PII-Synthetic (v6.0.0)** dataset on Hugging Face Hub!
* **[October 2026]** Launched the interactive **LINE Chat Simulator & Privacy Gateway** demo on [Hugging Face Spaces](https://huggingface.co/spaces/DuckerMaster/DuckerPII-Thai-Gateway).
* **[October 2026]** Achieved **98.33% OOD Recall** and dropped adversarial chat span leak by 3x (down to **1.67%**).

---

## 📖 Model Description

* **Model Type:** Token Classification / Named Entity Recognition (NER)
* **Underlying Architecture:** RoBERTa contextual subword encoder (WangchanBERTa-base fine-tuned)
* **Parameters:** ~110M
* **Primary Language:** Thai (`th`)
* **Context Length:** 416 tokens
* **Supported Entities (7 Classes):** `NAME`, `ID_CARD`, `PHONE`, `EMAIL`, `BANK_ACC`, `BANK`, `ADDRESS`
* **Training Corpus:** [DuckerMaster/Thai-PII-Synthetic v6.0.0](https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic) (20,000 sentences, 100% synthetic, zero real personal data)
* **License:** Apache 2.0 (Commercial Friendly)

---

## 📊 Benchmark Results

### 1. In-Domain & Out-of-Distribution (OOD) Performance

Evaluated against the held-out test split (2,000 sentences / 6,052 entities with 0% template and lexical overlap) and an adversarial OOD noisy chat benchmark:

| Metric | Regex Baseline | **DuckerPII-Thai-Base (Ours)** | **Gain vs. Baseline** |
| :--- | :---: | :---: | :---: |
| **In-Domain Macro Precision** | 0.7142 | **0.9989** | **+28.47%** |
| **In-Domain Macro Recall** | 0.6079 | **0.9989** | **+39.10%** |
| **In-Domain Macro F1-Score** | 0.6432 | **0.9989** | **+35.57%** |
| **In-Domain Micro F1-Score** | 0.5981 | **0.9989** | **+40.08%** |
| **OOD Noisy Chat Recall** | 0.4210 | **98.33%** | **+56.23%** |
| **OOD Span Leak Rate** | 57.90% | **1.67%** | **-56.23% (Leak Dropped)** |
| **Single-Core CPU Latency** | **< 1 ms** | **~42 ms/sample** | Real-time ready |

### 2. Class-by-Class F1 Breakdown

| Entity Label | Precision | Recall | F1-Score | Invariant / Robustness Resolved |
| :--- | :---: | :---: | :---: | :--- |
| **`NAME`** | 0.9989 | 0.9984 | **0.9986** | Resolves nicknames, transliterated names & unanchored contexts |
| **`ID_CARD`** | 1.0000 | 1.0000 | **1.0000** | Mod11 Checksum mathematical validation |
| **`PHONE`** | 0.9991 | 0.9991 | **0.9991** | Validated Thai telco prefixes (`08`, `09`, `06`, `02`) |
| **`EMAIL`** | 1.0000 | 1.0000 | **1.0000** | Fullmatch RFC compliance; eliminates alphanumeric tracking false positives |
| **`BANK_ACC`** | 0.9980 | 1.0000 | **0.9990** | Differentiates order IDs, tracking, and barcodes |
| **`BANK`** | 1.0000 | 0.9976 | **0.9988** | Formal bank names and colloquial abbreviations |
| **`ADDRESS`** | 0.9964 | 0.9978 | **0.9971** | Fragmented landmarks, subdistrict/district hierarchies |

### 3. Visual Dashboard
The complete 12-step validation harness executed on Kaggle demonstrates training stability and confusion matrix convergence:

<div align="center">
  <img src="reports/final_dashboard.png" alt="DuckerPII Performance Dashboard" width="850">
</div>

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/PlamGG/DuckerPII-Thai-Gateway.git
cd DuckerPII-Thai-Gateway
pip install -r requirements.txt
```

### Model Inference (Transformers)

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "DuckerMaster/DuckerPII-Thai-Base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.eval()

text = "แจ้งส่งพัสดุให้คุณ นพดล แซ่ลี้ โทร 0812345678 จัดส่งที่ 12/4 ซอยสุขุมวิท 21 คลองเตย กทม 10110 ครับ"

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=-1)[0]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred_idx in zip(tokens, predictions):
    label = model.config.id2label[pred_idx.item()]
    if label != "O":
        print(f"{token:<20} -> {label}")
```

### End-to-End De-Identification & Masking Pipeline

```python
from transformers import pipeline

ner_pipeline = pipeline(
    "token-classification",
    model="DuckerMaster/DuckerPII-Thai-Base",
    aggregation_strategy="simple"
)

raw_chat = "คุณกานดา จันทร์เพ็ญ โอนเงิน 500 บาท เข้าบัญชี 098-7-65432-1 ธนาคารกรุงไทย นะคะ"
results = ner_pipeline(raw_chat)

masked_text = raw_chat
for entity in sorted(results, key=lambda x: x["start"], reverse=True):
    masked_text = (
        masked_text[:entity["start"]]
        + f"<{entity['entity_group']}>"
        + masked_text[entity["end"]:]
    )

print("Original :", raw_chat)
print("Masked   :", masked_text)
# Output: <NAME> โอนเงิน 500 บาท เข้าบัญชี <BANK_ACC> <BANK> นะคะ
```

---

## 🏗️ System Architecture (ThaiPIIGateway)

DuckerPII adopts a **Two-Tier Cascaded Hybrid Architecture** combining the generalization capability of a deep neural transformer with deterministic safety gates:

```
[Raw Input / E-Commerce Chat]
              │
              ▼
 ┌────────────────────────────────────────────────────────┐
 │ Tier 1: Neural Representation (DuckerPII-Thai-Base)    │
 │ Subword Tokenization + RoBERTa Sequence Tagging        │
 └───────────────────────┬────────────────────────────────┘
                         │ Predicted Candidate Spans
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Tier 2: Deterministic Post-Processing Gate             │
 │  ├── Thai ID Card Mod11 Mathematical Verification      │
 │  ├── Telecommunication Prefix Validator (06, 08, 09)   │
 │  ├── RFC Fullmatch Email Filter                        │
 │  └── Alpha-Numeric Non-PII Filter (Tracking / Barcodes)│
 └───────────────────────┬────────────────────────────────┘
                         │ 100% Sanitized & Verified Spans
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Three Production Output Modes:                         │
 │  1. Placeholder Masking (<NAME>, <PHONE>)              │
 │  2. Synthetic Replacement (Realistic Anonymization)    │
 │  3. Reversible Audit Vault (Lossless JSON Map)         │
 └────────────────────────────────────────────────────────┘
```

---

## 🔒 Safety, Limitations & Ethical Use

* **100% Synthetic Data Invariant:** Trained exclusively on synthetic data created under Privacy Enhancing Technologies (PETs) guidelines. No personal identities of real living individuals were scraped or included.
* **Intended Use:** Designed strictly for **privacy protection, log masking, and pre-LLM data de-identification**.
* **Prohibited Use:** Strictly prohibited for unauthorized surveillance, de-anonymization attacks, harassment, or financial fraud.

---

## 👥 DuckerPII Team & Maintainers
Developed and maintained by **DuckerMaster**. For technical inquiries, enterprise integration guidelines, or vulnerability disclosures, please open an issue or pull request.

---

## 📜 Citation

If you find **DuckerPII** useful in your research or business applications, please cite:

```bibtex
@misc{duckermaster2026duckerpii,
  author       = {DuckerMaster},
  title        = {DuckerPII: High-Performance Thai PII De-Identification & Privacy Gateway},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/PlamGG/DuckerPII-Thai-Gateway}}
}
```
