#!/usr/bin/env python3
"""
Rename VibeVoice-ASR ONNX files from custom names to transformers.js v4 convention.

transformers.js v4 constructs file paths as: onnx/{fileName}{dtype_suffix}.onnx
For Seq2Seq with dtype="q4" (suffix "_q4"), it expects:
  - encoder_model_q4.onnx          (was: onnx/int4/speech_encoder.onnx)
  - decoder_model_merged_q4.onnx   (was: onnx/int4/decoder_with_speech.onnx)

The .onnx graph files contain internal protobuf references to their external data
file names (in external_data[location] fields). This script:
1. Downloads .onnx graph files from HuggingFace
2. Updates all external_data location references to the new names
3. Saves with the new filenames
4. Uploads everything to onnx/ on HuggingFace

Requirements:
    pip install onnx huggingface_hub

Usage:
    python rename_onnx.py
    # Then manually upload or use the --upload flag:
    python rename_onnx.py --upload
"""

import argparse
import os
import shutil
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
DOWNLOAD_DIR = Path("/tmp/vibevoice-split/downloads")
OUTPUT_DIR = Path("/tmp/vibevoice-split/output")

# Mapping: (subfolder, old_name) -> new_name (all go into onnx/)
FILE_RENAMES = {
    # Encoder
    "speech_encoder.onnx": "encoder_model_q4.onnx",
    "speech_encoder.onnx.data": "encoder_model_q4.onnx_data",
    # Decoder
    "decoder_with_speech.onnx": "decoder_model_merged_q4.onnx",
    "decoder_with_speech.onnx_data": "decoder_model_merged_q4.onnx_data",
    "decoder_with_speech.onnx_data_1": "decoder_model_merged_q4.onnx_data_1",
    "decoder_with_speech.onnx_data_2": "decoder_model_merged_q4.onnx_data_2",
    "decoder_with_speech.onnx_data_3": "decoder_model_merged_q4.onnx_data_3",
}

# Mapping for external_data location fields inside .onnx protobuf
# old_data_name -> new_data_name
DATA_RENAMES = {
    "speech_encoder.onnx.data": "encoder_model_q4.onnx_data",
    "decoder_with_speech.onnx_data": "decoder_model_merged_q4.onnx_data",
    "decoder_with_speech.onnx_data_1": "decoder_model_merged_q4.onnx_data_1",
    "decoder_with_speech.onnx_data_2": "decoder_model_merged_q4.onnx_data_2",
    "decoder_with_speech.onnx_data_3": "decoder_model_merged_q4.onnx_data_3",
}


def download_files():
    """Download all ONNX files from HuggingFace."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for old_name in FILE_RENAMES:
        print(f"Downloading onnx/int4/{old_name}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"onnx/int4/{old_name}",
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False,
        )
    print("All files downloaded.")


def update_onnx_external_data_refs(onnx_path: Path, output_path: Path):
    """
    Load an ONNX model, update external_data location references, and save.
    Uses onnx.load with load_external_data=False to avoid loading huge data files.
    """
    print(f"Updating external data references in {onnx_path.name}...")
    model = onnx.load(str(onnx_path), load_external_data=False)

    updated_count = 0
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value in DATA_RENAMES:
                old_val = entry.value
                entry.value = DATA_RENAMES[old_val]
                updated_count += 1
                print(f"  {old_val} -> {entry.value}")

    print(f"  Updated {updated_count} external data references.")
    onnx.save(model, str(output_path))
    print(f"  Saved to {output_path.name}")


def rename_files():
    """Rename all files and update ONNX protobuf references."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = DOWNLOAD_DIR / "onnx" / "int4"

    # First, process .onnx graph files (update internal references)
    for old_name, new_name in FILE_RENAMES.items():
        src = src_dir / old_name
        dst = OUTPUT_DIR / new_name

        if old_name.endswith(".onnx") and not old_name.endswith(".onnx.data"):
            # This is a graph file — update external data references
            update_onnx_external_data_refs(src, dst)
        else:
            # This is a data file — just copy with new name
            print(f"Copying {old_name} -> {new_name}")
            shutil.copy2(src, dst)

    print(f"\nAll files ready in {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


def upload_files():
    """Upload renamed files to HuggingFace onnx/ directory."""
    api = HfApi()

    for new_name in FILE_RENAMES.values():
        local_path = OUTPUT_DIR / new_name
        remote_path = f"onnx/{new_name}"
        print(f"Uploading {new_name} -> {remote_path}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
        )
    print("All files uploaded.")


def main():
    parser = argparse.ArgumentParser(description="Rename VibeVoice ONNX files for transformers.js v4")
    parser.add_argument("--upload", action="store_true", help="Upload renamed files to HuggingFace")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading (use existing files)")
    args = parser.parse_args()

    if not args.skip_download:
        download_files()

    rename_files()

    if args.upload:
        upload_files()
    else:
        print("\nTo upload, run again with --upload flag.")


if __name__ == "__main__":
    main()
