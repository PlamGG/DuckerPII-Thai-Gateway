---
title: DuckerPII Thai Gateway Demo
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Thai PII De-Identification Gateway with LINE Chat Simulator
---

# DuckerPII Thai Gateway Demo

Interactive LINE Chat Simulator and Enterprise Privacy De-Identification Gateway powered by **DuckerPII-Thai-Base**.

## Features
- **LINE Smartphone Chat Simulator:** Real-time conversational interface with quick test scenarios.
- **Three-Tier Output:**
  1. *De-Identified Text:* Clean token placeholders (`<NAME>`, `<PHONE>`) for LLM prompting.
  2. *Synthetic Replacement (Anonymization):* Replaces real PII with realistic synthetic Thai entities while maintaining contextual semantics.
  3. *Audit & Extraction Logs:* Character-level offsets, label classification, confidence scores, and latency metrics.
- **Hybrid Guard Pipeline:** Neural token classification + deterministic regex and checksum validation (Mod11 for ID Card, prefix filters for Phone).

Powered by [DuckerMaster/DuckerPII-Thai-Base](https://huggingface.co/DuckerMaster/DuckerPII-Thai-Base) and [DuckerMaster/Thai-PII-Synthetic](https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic).
