import sys
from agents import run_pipeline, read_docx_text
from docx import Document

# CLI entry point — reads the local resume file and runs the full pipeline

import os

RESUME_PATH = os.path.join(os.path.dirname(__file__), "Suyash_Jalan_Resume_Updated (3).docx")

sys.stdout.reconfigure(encoding="utf-8")

def progress(step: int, msg: str):
    print(f"[Step {step}] {msg}")

with open(RESUME_PATH, "rb") as f:
    file_bytes = f.read()

results = run_pipeline(file_bytes, progress_callback=progress)

print("\n" + "="*60)
print("EXTRACTED DATA")
print("="*60)
print(results["extract"])

print("\n" + "="*60)
print("SKILLS ANALYSIS")
print("="*60)
print(results["analysis"])

print("\n" + "="*60)
print("CAREER ROADMAP")
print("="*60)
print(results["advice"])