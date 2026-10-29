# Thai PII Privacy Gateway: Project Mission & Core Objectives
> **เอกสารนิยามจุดประสงค์และกรอบการทำงานของระบบ (Project Charter)**  
> *เอกสารภายในสำหรับกำหนดทิศทางการพัฒนา การประเมินผล และการนำเสนอผลงานเชิงวิศวกรรม*

---

## 1. วิสัยทัศน์และจุดประสงค์หลัก (Core Mission & Positioning)

### 1.1 พันธกิจ (Mission Statement)
สร้างโครงสร้างพื้นฐานด้านความปลอดภัยของข้อมูล (Privacy Middleware Layer) ภาษาไทย เพื่อทำหน้าที่เป็น **Zero-Trust Privacy Gateway** ระหว่างผู้ใช้งาน แอปพลิเคชันภายในองค์กร และผู้ให้บริการโมเดลภาษาขนาดใหญ่ภายนอก (External LLMs เช่น OpenAI, Claude, Gemini)

### 1.2 ปัญหาที่ระบบเข้ามาแก้ไข (Problem Statement)
* **ข้อจำกัดด้านกฎหมาย PDPA:** องค์กรในประเทศไทยมีความต้องการประยุกต์ใช้ Generative AI ในระบบบริการลูกค้าและงานเอกสาร แต่มีความเสี่ยงสูงต่อการทำข้อมูลส่วนบุคคล (PII) หลุดออกไปยัง Cloud Provider ภายนอก
* **ช่องโหว่ของโมเดล NER ทั่วไป:** โมเดล Deep Learning มักมีปัญหา False Positives กับตัวเลขทางธุรกิจ (เช่น รหัสสต็อก, เลขใบแจ้งหนี้) และขาดการตรวจสอบความถูกต้องเชิงอัลกอริทึม (Checksum) ทำให้ไม่ปลอดภัยเพียงพอสำหรับงาน Production
* **ข้อจำกัดเฉพาะของภาษาไทย:** ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ ไม่มีตัวพิมพ์เล็ก/ใหญ่ และมีคำนำหน้านามที่กำกวม ทำให้เครื่องมือ Masking สากล (เช่น Microsoft Presidio หรือ spaCy) ตรวจจับข้อมูลภาษาไทยได้ไม่มีประสิทธิภาพ

---

## 2. เสาหลักทางสถาปัตยกรรม (Core Architectural Pillars)

```
                 [ User Input / Client App ]
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │       THAI PII CASCASED GATEWAY           │
        │                                           │
        │  [Tier 1] Deterministic Rule Engine       │
        │   - Lookaround Regex (Boundary-safe)      │
        │   - Mathematical Mod11 ID Card Checksum   │
        │                                           │
        │  [Tier 2] Neural Contextual Extractor     │
        │   - WangchanBERTa (Subword-aware)         │
        │   - Free-form Name & Address NER          │
        │                                           │
        │  [Sanitization & Disambiguation]          │
        │   - Honorific / Preposition Normalization │
        │   - Acronym Conflict Resolver             │
        └─────────────────────┬─────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [ Masked Prompt ]             [ Secure Token Map ]
               │                       (เก็บในหน่วยความจำเฉพาะ Session)
               ▼
       [ External LLM ]
               │
               ▼
       [ LLM Output ]
               │
               ▼
     [ Unmasking Engine ] ◄──────────┘
               │
               ▼
   [ Final Response to User ]
```

### เสาหลักที่ 1: Two-Tier Cascaded Hybrid Architecture
* **Tier 1 (Deterministic Engine):** จับข้อมูลที่มีโครงสร้างและรูปแบบแน่นอน (เลขบัตรประชาชน, เบอร์โทร, บัญชีธนาคาร, อีเมล) ด้วย Regular Expression และสูตรคณิตศาสตร์ Mod11 ตรวจสอบความถูกต้อง ทำงานทันทีโดยมีความเร็วสูง
* **Tier 2 (Neural Transformer):** ใช้ `WangchanBERTa` สกัดข้อมูลที่ต้องอาศัยบริบทและโครงสร้างประโยคอิสระ (ชื่อ-นามสกุล, ที่อยู่ภาษาไทย)
* **Disambiguation Layer:** คัดกรองและจัดลำดับความสำคัญของข้อมูล ป้องกันไม่ให้รหัสคำสั่งซื้อหรือตัวเลขทางธุรกิจถูกเข้าใจผิดเป็นข้อมูลส่วนบุคคล

### เสาหลักที่ 2: Reversible In-Memory Masking Engine
* แปลง PII ทุกรายการเป็น Label Placeholder เฉพาะตัว เช่น `<NAME_1>`, `<PHONE_1>`, `<ID_CARD_1>`
* รักษา Context ความหมายของประโยคเพื่อให้ LLM เข้าใจความสัมพันธ์และสร้างคำตอบได้ถูกต้อง
* ถอดรหัสคืนรูป (Unmask) อัตโนมัติในฝั่งขาออก โดยข้อมูล Mapping จะอยู่ในหน่วยความจำชั่วคราวและถูกทำลายหลังเสร็จสิ้น Request

### เสาหลักที่ 3: Zero Real-World PII (100% Synthetic Standard)
* ข้อมูลทั้งหมดที่ใช้ในการพัฒนา ฝึกสอน และประเมินผลระบบ ต้องเป็น **ข้อมูลสังเคราะห์ (Synthetic Data) 100%**
* ดำเนินการตามแนวปฏิบัติการคุ้มครองข้อมูลส่วนบุคคลสำหรับการพัฒนา AI ของ สคส. (PDPC) จัดเป็นเทคโนโลยีเสริมสร้างความเป็นส่วนตัว (PETs) ตั้งแต่กระบวนการออกแบบ (Privacy-by-Design)

---

## 3. ดัชนีชี้วัดความสำเร็จทางวิศวกรรม (Engineering Target & SLA)

ระบบถูกออกแบบโดยยึดเกณฑ์การประเมินทางวิศวกรรมระดับ Enterprise-grade:

| ตัวชี้วัด (Metric) | เกณฑ์เป้าหมาย (SLA) | ความสำคัญ |
| :--- | :--- | :--- |
| **Inference Latency (p95)** | **< 25 ms / sentence** | ป้องกันไม่ให้ Gateway กลายเป็นคอขวดของระบบแชตและ API |
| **Throughput (GPU Batching)** | **≥ 50 samples / sec** | รองรับการประมวลผลพร้อมกันในระบบ Customer Service ขนาดใหญ่ |
| **Tier-1 Critical PII Leak Rate** | **< 0.1%** (`ID_CARD`, `BANK_ACC`) | ความปลอดภัยขั้นเด็ดขาดต่อข้อมูลระบุตัวตนและข้อมูลทางการเงิน |
| **Overall PII Recall (F5-Score)** | **≥ 99.0%** ($\beta = 5$) | ให้น้ำหนักกับการป้องกันข้อมูลหลุด (Recall) มากกว่าความแม่นยำทั่วไป |
| **Out-of-Distribution Recall** | **≥ 95.0%** (Noisy Chat) | คงความสามารถในการตรวจจับได้แม้ผู้ใช้พิมพ์ผิดหรือใช้คำสแลง |
| **Round-trip Integrity** | **100% Exact Match** | ข้อความที่ผ่านการ Mask และ Unmask ต้องคืนรูปสมบูรณ์ ไม่สูญหาย |

---

## 4. สถานการณ์นำไปใช้งานจริง (Target Use Cases)

1. **Enterprise Customer Support Bot:** คัดกรองและเซ็นเซอร์ข้อความของลูกค้าก่อนส่งไปยัง LLM ในการตอบคำถาม ช่วยให้พนักงานและระบบอัตโนมัติทำงานได้โดยไม่ต้องสัมผัส PII ดิบ
2. **Enterprise RAG / Knowledge Base Sanitizer:** ทำความสะอาดข้อมูลแชตเก่า, บันทึกการสนทนา, และเอกสารภายใน ก่อนนำเข้าสู่ Vector Database
3. **Data Anonymization Pipeline for BI & Analytics:** แปลงชุดข้อมูลการติดต่อและธุรกรรมให้อยู่ในรูปนิรนาม เพื่อให้ทีมนักวิเคราะห์ข้อมูลใช้งานได้โดยไม่ขัดต่อกฎหมาย PDPA

---

## 5. จุดแตกต่างที่ใช้ในการนำเสนอ (Key Differentiators for Portfolio)

* **ไม่ใช่แค่โมเดล แต่เป็นระบบ Middleware ครบวงจร:** มีทั้งขั้นตอนการตรวจจับ (Detection), การเซ็นเซอร์แบบถอดรหัสได้ (Masking), และการคืนรูป (Unmasking)
* **แก้จุดอ่อนของ Deep Learning ด้วย Domain Logic:** ผสานสูตรคณิตศาสตร์ Mod11 สำหรับเลขบัตรประชาชนไทย เพื่อแยกแยะบัตรจริงออกจากรหัสสินค้า 13 หลักได้อย่างเด็ดขาด
* **Benchmark โปร่งใส 4 มิติ:** มีผลการทดสอบเปรียบเทียบชัดเจนระหว่าง Regex Only, Neural Only, และ Hybrid Architecture พิสูจน์ให้เห็นถึงความคุ้มค่าของการทำ Two-Tier System
