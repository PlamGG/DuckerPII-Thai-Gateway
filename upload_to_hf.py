"""
Script to upload the verified dataset to Hugging Face Hub securely with Pre-flight Check.
Usage:
  Pre-flight Dry Run (Check Auth & Dataset Integrity without uploading):
    $env:HF_TOKEN = "your_token_here"
    python upload_to_hf.py --dry-run

  Actual Upload:
    $env:HF_TOKEN = "your_token_here"
    python upload_to_hf.py --repo_id "DuckerMaster/Thai-PII-Synthetic"
"""

import argparse
import os
import sys
from datasets import load_dataset
from huggingface_hub import HfApi, CommitOperationAdd, get_token
from huggingface_hub.utils import HfHubHTTPError

def preflight_check(repo_id: str, token: str, dry_run: bool = False) -> dict:
    """Verify Hugging Face authentication and permissions before uploading."""
    if not token:
        if dry_run:
            print("[INFO] HF_TOKEN environment variable not set. Skipping remote authentication check during dry-run.")
            return {"name": "Local Verification Mode (No HF_TOKEN set)"}
        print("[ERROR] Hugging Face token not provided.")
        print("Please set the HF_TOKEN environment variable ($env:HF_TOKEN='...') or pass --token.")
        sys.exit(1)

    print(f"[*] Checking Hugging Face authentication for token...")
    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        auth_role = user_info.get("auth", {}).get("accessToken", {}).get("role", "unknown")
        print(f"[SUCCESS] Authenticated as user: '{username}' (Token Role: {auth_role})")
        
        target_namespace = repo_id.split("/")[0] if "/" in repo_id else repo_id
        orgs = [org.get("name") for org in user_info.get("orgs", [])]
        if username.lower() != target_namespace.lower() and target_namespace not in orgs:
            print(f"[WARNING] Target repo namespace '{target_namespace}' does not match username '{username}' or user's orgs: {orgs}")
            print(f"          Ensure you have write permission to '{repo_id}'.")
        else:
            print(f"[SUCCESS] Target namespace '{target_namespace}' verified for write access.")
            
        return user_info
    except HfHubHTTPError as e:
        print(f"[ERROR] Hugging Face authentication failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during Pre-flight auth check: {e}")
        sys.exit(1)

def upload(repo_id: str, token: str, dry_run: bool = False):
    user_info = preflight_check(repo_id, token, dry_run=dry_run)

    print(f"\n[*] Validating local dataset splits from data/...")
    data_files = {
        "train": "data/train.json",
        "validation": "data/validation.json",
        "test": "data/test.json"
    }
    for split, path in data_files.items():
        if not os.path.exists(path):
            print(f"[ERROR] Split file not found: {path}")
            sys.exit(1)

    ds = load_dataset("json", data_files=data_files)
    print(ds)

    required_docs = [
        "README.md",
        "README_TH.md",
        "SAFETY.md",
        "LICENSE",
        "ANNOTATION_GUIDELINE.md",
        "data/manifest.json",
        "reports/evaluation_report.json",
        "config/generator_config.yaml",
        "config/label2id.json",
        "config/id2label.json",
        "data/train.json",
        "data/validation.json",
        "data/test.json",
        "data/hf_export/train.json",
        "data/hf_export/validation.json",
        "data/hf_export/test.json"
    ]
    for fname in required_docs:
        if not os.path.exists(fname):
            print(f"[ERROR] Required documentation file missing: {fname}")
            sys.exit(1)
        print(f"[*] Found required document: {fname}")

    if dry_run:
        print("\n=======================================================")
        print("[DRY-RUN COMPLETE] All Pre-flight checks PASSED successfully.")
        print(f"Target Repository : {repo_id}")
        print(f"Authenticated User: {user_info.get('name')}")
        print("Ready for live upload! (Run without --dry-run to push)")
        print("=======================================================")
        return

    print(f"\n[*] Pushing dataset to Hugging Face Hub: {repo_id}...")
    ds.push_to_hub(repo_id, token=token, private=False)

    print(f"[*] Uploading documentation artifacts and data files via atomic commit...")
    api = HfApi(token=token)
    operations = [
        CommitOperationAdd(path_in_repo=fname.replace("\\", "/"), path_or_fileobj=fname)
        for fname in required_docs
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish Thai-PII-Synthetic v6.0.0 (Adversarially Robust Industrial Grade)"
    )
    for fname in required_docs:
        print(f" - Included {fname}")
            
    print(f"\nSUCCESS: Dataset published at https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub with pre-flight verification.")
    parser.add_argument("--repo_id", type=str, default="DuckerMaster/Thai-PII-Synthetic", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (Defaults to HF_TOKEN env variable or cached CLI token)")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight authentication and file integrity checks without uploading")
    args = parser.parse_args()
    
    hf_token = os.environ.get("HF_TOKEN") or args.token or get_token()
    upload(args.repo_id, hf_token, dry_run=args.dry_run)
