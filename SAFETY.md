# Safety and Misuse Policy: Thai PII Synthetic Dataset

## 1. Overview & Statement of Intent
**Thai PII Synthetic Dataset** (`DuckerMaster/Thai-PII-Synthetic`) is an open-source, 100% synthetic research benchmark created for training and evaluating Thai Named Entity Recognition (NER) and token classification models for privacy compliance and data de-identification (Masking/Redaction) under the **Thailand Personal Data Protection Act B.E. 2562 (PDPA)** and referencing the **Draft PDPC AI Guidelines**.

---

## 2. Intended Users & Use Cases
### Authorized Users:
* Natural Language Processing (NLP) researchers and engineers.
* Privacy and compliance teams developing data sanitization tools.
* Quality Assurance (QA) engineers needing test data without real customer records.

### Approved Use Cases:
* **Pre-LLM Privacy Gateways:** Detecting and redacting personal data before sending prompts to external APIs.
* **Customer Support Chat Anonymization:** Masking sensitive entities in transcripts before storing in data lakes.
* **NER Benchmarking:** Evaluating token classification models on complex Thai syntax and address structures.

---

## 3. Prohibited Use Cases
Users and organizations are **strictly prohibited** from utilizing this dataset or derived artifacts for:
1. **Phishing, Scams, or Social Engineering:** Crafting deceptive messages, fake invoices, or fraudulent payment slips.
2. **Surveillance & Doxxing:** Constructing automated tools aimed at unmasking or tracking private individuals.
3. **Synthetic Identity Fraud:** Attempting to register accounts or bypass KYC systems using synthetic Mod11 IDs.
4. **Data Contamination:** Uploading real personal data of real individuals into this public repository or public issue trackers.

---

## 4. Re-Identification & Misuse Risks
* **Combinatorial Safety:** Every sample is generated using disjoint, randomly combined vocabulary pools. No sample represents an actual living person.
* **Realistic Formatting Warning:** Because phone prefixes (`08`, `09`, `06`), bank formats, and Mod11 checksums follow official Thai syntactic standards, numbers may coincidentally match real-world formats. They **must never be treated as verified contact details**.

---

## 5. Responsible Deployment Guidelines
When deploying models trained on this dataset:
* **Apply Human-in-the-Loop:** For high-stakes decisions (e.g. legal or medical redaction), combine automated NER with human spot-checks.
* **Implement Defense-in-Depth:** Use cascading architectures (Rule-based regex + Deep Transformer) to balance speed and accuracy.
* **Do Not Claim Complete PDPA Immunity:** This dataset covers 7 common PII types; it does not cover all possible personal data under PDPA.

---

## 6. Incident Reporting & Vulnerability Disclosure
If you discover unintended real-world entity collisions, security flaws, or evidence of dataset misuse, please report immediately to:
* **Project Maintainers:** Open a private security advisory on the GitHub repository or contact via Hugging Face Community tab.
* Response SLA: Reports will be acknowledged within 48 hours, and any colliding synthetic patterns will be purged in the subsequent minor patch.
