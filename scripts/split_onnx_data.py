"""
Split large ONNX external data files into ~2 GB shards.

This is needed because WebGPU/browser fetch can't reliably handle
single files >2 GB. Transformers.js v4 supports loading sharded
external data when configured via transformers.js_config in config.json.

Usage:
    python scripts/split_onnx_data.py /path/to/onnx/int4/

This will:
  1. Split decoder_with_speech.onnx.data into ~2 GB shards
  2. Update decoder_with_speech.onnx to reference the shards
  3. Leave speech_encoder files unchanged (under 2 GB)

Requirements:
    pip install onnx
"""

import argparse
import os
import sys

import onnx
from onnx.external_data_helper import (
    convert_model_to_external_data,
    write_external_data_tensors,
)


SHARD_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def split_model_data(model_path: str) -> None:
    """Re-save an ONNX model with external data split into shards."""
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    data_path = model_path + ".data"
    if not os.path.exists(data_path):
        print(f"ERROR: External data file {data_path} not found")
        sys.exit(1)

    data_size = os.path.getsize(data_path)
    print(f"Loading {model_path} (external data: {data_size / 1e9:.2f} GB)...")

    # Load the ONNX model (loads graph only, external data stays on disk)
    model = onnx.load(model_path, load_external_data=True)

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    base_name = model_name  # e.g. "decoder_with_speech.onnx"

    # Remove old data file before re-saving
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"Removed old {data_path}")

    # Convert to external data with per-tensor files (one file per tensor)
    # Then ORT will name them as <location>_0, <location>_1, etc.
    # Actually, onnx library's convert_model_to_external_data with
    # all_tensors_to_one_file=False creates one file per tensor.
    # For sharding, we use size_threshold and all_tensors_to_one_file=True
    # with a specific filename, then manually split.

    # Approach: Re-save with external data in a single file, then split manually
    external_data_name = f"{base_name}_data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=0,
        convert_attribute=True,
    )

    # Save the model graph (without data) and write external data
    print(f"Writing external data to {external_data_name}...")
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=0,
        convert_attribute=True,
    )

    merged_data_path = os.path.join(model_dir, external_data_name)
    merged_size = os.path.getsize(merged_data_path)
    print(f"Merged data file: {merged_size / 1e9:.2f} GB")

    if merged_size <= SHARD_SIZE_BYTES:
        print("Data file is under 2 GB, no splitting needed.")
        return

    # Split the merged data file into shards
    num_shards = (merged_size + SHARD_SIZE_BYTES - 1) // SHARD_SIZE_BYTES
    print(f"Splitting into {num_shards} shards of ~{SHARD_SIZE_BYTES / 1e9:.1f} GB...")

    shard_paths = []
    with open(merged_data_path, "rb") as f:
        for i in range(num_shards):
            if i == 0:
                shard_name = external_data_name
                shard_path = merged_data_path
            else:
                shard_name = f"{external_data_name}_{i}"
                shard_path = os.path.join(model_dir, shard_name)

            chunk = f.read(SHARD_SIZE_BYTES)
            if i > 0:
                # For shard 0, the file already exists with full data
                # We'll rewrite it after reading all shards
                with open(shard_path, "wb") as sf:
                    sf.write(chunk)

            shard_paths.append((shard_name, len(chunk)))
            print(f"  Shard {i}: {shard_name} ({len(chunk) / 1e9:.2f} GB)")

    # Rewrite shard 0 (truncate the merged file to SHARD_SIZE_BYTES)
    with open(merged_data_path, "r+b") as f:
        f.seek(0)
        first_shard_data = f.read(SHARD_SIZE_BYTES)
        f.seek(0)
        f.write(first_shard_data)
        f.truncate()

    print(f"\nDone! Created {num_shards} shard files.")
    print(f"Number of shards for transformers.js_config: {num_shards}")

    # Print summary
    print("\nFiles created:")
    for name, size in shard_paths:
        path = os.path.join(model_dir, name)
        actual_size = os.path.getsize(path)
        print(f"  {name}: {actual_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Split ONNX external data files into ~2 GB shards"
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing the ONNX models (e.g., onnx/int4/)",
    )
    parser.add_argument(
        "--model",
        default="decoder_with_speech.onnx",
        help="ONNX model filename to split (default: decoder_with_speech.onnx)",
    )
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model)
    split_model_data(model_path)


if __name__ == "__main__":
    main()
