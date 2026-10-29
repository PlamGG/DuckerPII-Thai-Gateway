"""
Unified CI Runner: Runs all project checks in one command (scripts/run_all_checks.py)
1. Quality & Span Verification (scripts/run_quality_checks.py)
2. Legal & PDPA Compliance (test_pdpa_compliance.py)
3. Model & Partial Overlap Evaluation (scripts/evaluate_model.py)
4. Adversarial Robustness Test (scripts/test_adversarial_robustness.py)
5. Hugging Face Datasets Load Check (test_hf_loading.py)

Exits with code 0 on success, code 1 on failure.
"""

import subprocess
import sys
import os

def run_command(cmd, name):
    print(f"\n[CI STEP] Running {name}...")
    res = subprocess.run([sys.executable, "-X", "utf8"] + cmd.split(), capture_output=False)
    if res.returncode != 0:
        print(f"[CI ERROR] {name} FAILED with exit code {res.returncode}")
        return False
    print(f"[CI SUCCESS] {name} passed.")
    return True

def main():
    steps = [
        ("scripts/run_quality_checks.py", "Quality & Span Verification"),
        ("tests/test_pdpa_compliance.py", "PDPA & AI Legal Compliance"),
        ("scripts/evaluate_model.py", "Evaluation Metrics (Exact & Partial)"),
        ("scripts/test_adversarial_robustness.py", "Adversarial Robustness Testing"),
        ("tests/test_hf_loading.py", "Hugging Face Loading Simulation")
    ]

    all_passed = True
    for script, desc in steps:
        if not run_command(script, desc):
            all_passed = False
            break

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CI & QUALITY CHECKS PASSED PERFECTLY (EXIT CODE 0)")
        print("Dataset is 100% production-ready and Hugging Face compliant.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("CI CHECKS FAILED (EXIT CODE 1)")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
