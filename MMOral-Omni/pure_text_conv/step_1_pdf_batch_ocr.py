#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
import time

# Configuration
BASE_PATH = "/data/user/workspace/document_processing/source_files"
OUTPUT_DIR = "processed_output"

# PDF file list
pdf_files = [
    "document_001.pdf",
    "document_002.pdf", 
    "document_003.pdf",
    "document_004.pdf",
    "document_005.pdf",
    "document_006.pdf",
]

def process_pdf(pdf_file):
    """Process single PDF file"""
    full_path = os.path.join(BASE_PATH, pdf_file)
    
    print(f"Processing: {pdf_file}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    cmd = [
        "mineru",
        "-p", full_path,
        "--output", OUTPUT_DIR,
        "--ocr",
        "--lang", "en",
        "--device", "cuda:0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {pdf_file} processing completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {pdf_file} processing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ {pdf_file} exception occurred: {e}")
        return False
    finally:
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    total_files = len(pdf_files)
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{total_files}] Starting processing...")
        
        if process_pdf(pdf_file):
            successful += 1
        else:
            failed += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Processing summary:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()
