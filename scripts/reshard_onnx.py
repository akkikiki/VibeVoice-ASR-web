#!/usr/bin/env python3
"""
Re-shard ONNX external data files so no single file exceeds the browser's
ArrayBuffer limit (~2 GB). Browsers cannot allocate a Uint8Array > 2^31 - 1 bytes.

This script:
1. Loads the ONNX model protobuf (without loading external data into memory)
2. Reads tensor data from original external data files
3. Writes new, smaller shard files (max 1.9 GB each)
4. Updates protobuf external_data references (location, offset)
5. Saves the updated ONNX model
6. Optionally uploads to HuggingFace

Requirements:
    pip install onnx huggingface_hub

Usage:
    python reshard_onnx.py
    python reshard_onnx.py --upload
"""

import argparse
import os
from pathlib import Path

import onnx

MAX_SHARD_SIZE = 1_900_000_000  # 1.9 GB — safely under browser's 2 GB ArrayBuffer limit

REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
INPUT_DIR = Path("/tmp/vibevoice-split/output")  # From previous rename step
OUTPUT_DIR = Path("/tmp/vibevoice-split/resharded")

# Only the decoder needs re-sharding (encoder is ~710 MB, well under limit)
DECODER_ONNX = "decoder_model_merged_q4.onnx"


def get_external_data_info(initializer):
    """Extract external data info from an ONNX initializer."""
    info = {}
    for entry in initializer.external_data:
        info[entry.key] = entry.value
    return info


def set_external_data_info(initializer, location, offset, length):
    """Update external data references on an ONNX initializer."""
    for entry in initializer.external_data:
        if entry.key == "location":
            entry.value = location
        elif entry.key == "offset":
            entry.value = str(offset)
    # Ensure offset entry exists
    keys = {e.key for e in initializer.external_data}
    if "offset" not in keys:
        initializer.external_data.add(key="offset", value=str(offset))


def reshard_decoder():
    """Re-shard the decoder's external data into files < MAX_SHARD_SIZE."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = INPUT_DIR / DECODER_ONNX
    print(f"Loading {onnx_path} (protobuf only, no external data)...")
    model = onnx.load(str(onnx_path), load_external_data=False)

    # Collect all tensors with external data, grouped by source file
    tensors = []
    for init in model.graph.initializer:
        info = get_external_data_info(init)
        if "location" not in info:
            continue
        tensors.append({
            "init": init,
            "location": info["location"],
            "offset": int(info.get("offset", "0")),
            "length": int(info["length"]),
        })

    print(f"Found {len(tensors)} tensors with external data")

    # Sort by (source file, offset) to maintain data locality
    tensors.sort(key=lambda t: (t["location"], t["offset"]))

    # Open all source data files
    src_files = {}
    for t in tensors:
        loc = t["location"]
        if loc not in src_files:
            src_path = INPUT_DIR / loc
            src_files[loc] = open(src_path, "rb")
            size_mb = src_path.stat().st_size / (1024 * 1024)
            print(f"  Source: {loc} ({size_mb:.1f} MB)")

    # Greedy bin-packing into new shards
    base_name = DECODER_ONNX  # e.g., "decoder_model_merged_q4.onnx"
    shard_idx = 0
    shard_offset = 0
    shard_file = None
    shard_path = None
    shard_assignments = []  # (tensor_idx, shard_name, new_offset)

    for i, t in enumerate(tensors):
        data_len = t["length"]

        # Start new shard if current one would exceed limit
        if shard_file is None or (shard_offset + data_len > MAX_SHARD_SIZE and shard_offset > 0):
            if shard_file is not None:
                print(f"  Shard {shard_path.name}: {shard_offset / (1024**2):.1f} MB")
                shard_file.close()
            shard_idx += 1 if shard_file is not None else 0
            suffix = "" if shard_idx == 0 else f"_{shard_idx}"
            shard_name = f"{base_name}_data{suffix}"
            shard_path = OUTPUT_DIR / shard_name
            shard_file = open(shard_path, "wb")
            shard_offset = 0

        # Read tensor data from source file
        src_f = src_files[t["location"]]
        src_f.seek(t["offset"])
        data = src_f.read(data_len)
        assert len(data) == data_len, f"Short read: expected {data_len}, got {len(data)}"

        # Write to new shard
        new_offset = shard_offset
        shard_file.write(data)
        shard_offset += data_len

        # Record assignment
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        current_shard_name = f"{base_name}_data{suffix}"
        shard_assignments.append((i, current_shard_name, new_offset))

    # Close last shard
    if shard_file is not None:
        print(f"  Shard {shard_path.name}: {shard_offset / (1024**2):.1f} MB")
        shard_file.close()

    num_shards = shard_idx + 1
    print(f"\nCreated {num_shards} shards")

    # Close source files
    for f in src_files.values():
        f.close()

    # Update protobuf references
    for i, shard_name, new_offset in shard_assignments:
        t = tensors[i]
        set_external_data_info(t["init"], shard_name, new_offset, t["length"])

    # Save updated ONNX model
    output_onnx = OUTPUT_DIR / DECODER_ONNX
    onnx.save(model, str(output_onnx))
    print(f"Saved updated {output_onnx.name}")

    # Also copy the encoder files (unchanged)
    import shutil
    for name in ["encoder_model_q4.onnx", "encoder_model_q4.onnx_data"]:
        src = INPUT_DIR / name
        dst = OUTPUT_DIR / name
        if src.exists() and not dst.exists():
            print(f"Copying {name} (unchanged)")
            shutil.copy2(src, dst)

    # Print summary
    print(f"\nAll files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    return num_shards


def upload_files():
    """Upload resharded files to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi()

    for f in sorted(OUTPUT_DIR.iterdir()):
        remote_path = f"onnx/{f.name}"
        print(f"Uploading {f.name} -> {remote_path}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
        )
    print("All files uploaded.")


def main():
    parser = argparse.ArgumentParser(description="Re-shard ONNX external data for browser compatibility")
    parser.add_argument("--upload", action="store_true", help="Upload resharded files to HuggingFace")
    args = parser.parse_args()

    num_shards = reshard_decoder()
    print(f"\n*** Update use_external_data_format in worker.js: decoder_model_merged: {num_shards} ***")

    if args.upload:
        upload_files()
    else:
        print("\nTo upload, run again with --upload flag.")


if __name__ == "__main__":
    main()
