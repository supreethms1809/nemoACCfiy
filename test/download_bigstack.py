import os
import boto3
from smart_open import open
from datasets import load_dataset
from pathlib import Path

# Programming languages to download
programming_languages = ["C", "C++", "Python", "Cuda"]

# Initialize AWS session for S3 access (anonymous access to public bucket)
from botocore import UNSIGNED
from botocore.config import Config

session = boto3.Session()
s3 = session.client("s3", 
                   config=Config(signature_version=UNSIGNED),
                   region_name="us-east-1")

def download_contents(blob_id: str, src_encoding: str) -> str:
    """Download file contents from Software Heritage S3 bucket."""
    try:
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        # Use unsigned client for anonymous access
        transport_params = {"client": s3}
        with open(s3_url, "rb", compression=".gz", transport_params=transport_params) as fin:
            content = fin.read()
            
        # Try to decode with the specified encoding, fallback to utf-8 with error handling
        try:
            return content.decode(src_encoding)
        except (UnicodeDecodeError, LookupError):
            return content.decode('utf-8', errors='replace')
            
    except Exception as e:
        print(f"Failed to download blob {blob_id}: {str(e)}")
        return ""

def check_dataset_exists(cache_path: str) -> bool:
    """Check if dataset already exists in cache."""
    cache_dir = Path(cache_path)
    return cache_dir.exists() and (cache_dir / "dataset_info.json").exists()

def filter_missing_content(ds, programming_language: str):
    """Filter dataset to only include samples that don't have content yet."""
    def has_content(row):
        # Check if 'text' field exists and is not empty
        return 'text' in row and row['text'] and row['text'].strip()
    
    # Filter out samples that already have content
    filtered_ds = ds.filter(has_content, desc=f"Filtering {programming_language} for missing content")
    missing_count = len(ds) - len(filtered_ds)
    
    if missing_count > 0:
        print(f"Found {missing_count} samples already with content, {len(filtered_ds)} remaining to download")
        return filtered_ds
    else:
        print(f"All {len(ds)} samples need content download")
        return ds

# Main execution
print("Starting BigCode Stack v2 dataset download...")

# No AWS credentials needed for public Software Heritage bucket

for programming_language in programming_languages:
    print(f"\nProcessing {programming_language}...")
    
    try:
        # Define cache path
        cache_path = os.path.expanduser(f"~/.cache/huggingface/datasets/bigcode_the-stack-v2-dedup_{programming_language}_with_content")
        
        # Check if dataset already exists
        if check_dataset_exists(cache_path):
            print(f"Dataset already exists in cache: {cache_path}")
            print("Loading existing dataset...")
            ds_with_content = load_dataset("bigcode/the-stack-v2-dedup", programming_language, split="train")
            # Try to load from cache if it exists
            try:
                from datasets import Dataset
                ds_with_content = Dataset.load_from_disk(cache_path)
                print(f"Loaded existing dataset: {len(ds_with_content)} samples")
            except:
                print("Could not load from cache, will re-download...")
                ds_with_content = None
        else:
            ds_with_content = None
        
        # Load the base dataset
        ds = load_dataset("bigcode/the-stack-v2-dedup", programming_language, split="train")
        print(f"Base dataset loaded: {len(ds)} samples for {programming_language}")
        
        # If we have an existing dataset, filter for missing content
        if ds_with_content is not None:
            # Filter to only download missing content
            ds_to_process = filter_missing_content(ds, programming_language)
            if len(ds_to_process) == 0:
                print(f"All content already downloaded for {programming_language}, skipping...")
                continue
        else:
            ds_to_process = ds
        
        # Download contents and add to dataset
        print(f"Downloading file contents for {programming_language}...")
        
        def add_content(row):
            content = download_contents(row["blob_id"], row["src_encoding"])
            row["text"] = content
            return row
        
        # Process with progress bar (using multiprocessing for faster downloads)
        ds_processed = ds_to_process.map(
            add_content,
            desc=f"Downloading {programming_language} contents",
            num_proc=32  # Use 32 processes for faster downloads
        )
        
        # If we had existing content, merge with new content
        if ds_with_content is not None:
            # Combine existing and new datasets
            from datasets import concatenate_datasets
            # Filter existing dataset to only include samples without content
            existing_without_content = ds_with_content.filter(
                lambda row: 'text' not in row or not row['text'] or not row['text'].strip()
            )
            # Combine with newly processed data
            ds_with_content = concatenate_datasets([existing_without_content, ds_processed])
        else:
            ds_with_content = ds_processed
        
        # Save to Hugging Face cache
        ds_with_content.save_to_disk(cache_path)
        print(f"Dataset saved to cache: {cache_path}")
        print(f"Successfully processed {programming_language}: {len(ds_with_content)} samples")
        
    except Exception as e:
        print(f"Failed to process {programming_language}: {str(e)}")
        continue

print("\nCompleted processing all datasets!")