"""
Verification script: loads the local JSON splits using Hugging Face datasets library
Simulates the exact behavior before pushing to Hugging Face Hub.
"""
from datasets import load_dataset
import os

def test_load():
    data_files = {
        "train": "data/train.json",
        "validation": "data/validation.json",
        "test": "data/test.json"
    }
    ds = load_dataset("json", data_files=data_files)
    print("=== Hugging Face Datasets Load Check ===")
    print(ds)
    print("\nFirst sample of train split:")
    print(ds["train"][0])
    import json
    with open("data/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(ds["train"]) == manifest["splits"]["train"]["samples"]
    assert len(ds["validation"]) == manifest["splits"]["validation"]["samples"]
    assert len(ds["test"]) == manifest["splits"]["test"]["samples"]
    print("\nALL SPLITS LOADED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_load()
