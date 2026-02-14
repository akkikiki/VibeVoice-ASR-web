#!/usr/bin/env python3
"""
Convert the embed_tokens weight from fp32 to fp16 to fit within browser's
2 GB ArrayBuffer limit, then re-shard all decoder external data.

The language_model.embed_tokens.weight tensor is 152064 * 3584 * fp32 = 2.03 GB,
which exceeds the browser's ArrayBuffer max size (~2 GB). Converting to fp16
reduces it to ~1.01 GB.

A Cast node is inserted after the Gather to convert the fp16 embedding back to
fp32 before it reaches the Concat node (which also receives fp32 speech_embeddings).

Steps:
1. Load decoder ONNX model (protobuf only)
2. Read embed_tokens fp32 data, convert to fp16
3. Update protobuf: change data_type, add Cast node
4. Re-shard all external data so every file < 1.9 GB
5. Save updated model + new shard files
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MAX_SHARD_SIZE = 1_900_000_000  # 1.9 GB
REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
INPUT_DIR = Path("/tmp/vibevoice-split/output")
OUTPUT_DIR = Path("/tmp/vibevoice-split/resharded")

DECODER_ONNX = "decoder_model_merged_q4.onnx"
EMBED_TENSOR_NAME = "language_model.embed_tokens.weight"


def convert_embedding_to_fp16(model):
    """
    Convert embed_tokens.weight from fp32 to fp16 and insert a Cast node.
    Returns the fp16 data as bytes and the updated model.
    """
    # Find the embedding initializer
    embed_init = None
    for init in model.graph.initializer:
        if init.name == EMBED_TENSOR_NAME:
            embed_init = init
            break
    assert embed_init is not None, f"Could not find {EMBED_TENSOR_NAME}"

    info = {e.key: e.value for e in embed_init.external_data}
    src_file = INPUT_DIR / info["location"]
    src_offset = int(info.get("offset", "0"))
    src_length = int(info["length"])

    print(f"Reading {EMBED_TENSOR_NAME} from {info['location']} ({src_length / (1024**3):.2f} GB)...")
    with open(src_file, "rb") as f:
        f.seek(src_offset)
        fp32_data = f.read(src_length)

    # Convert fp32 -> fp16
    print("Converting fp32 -> fp16...")
    fp32_array = np.frombuffer(fp32_data, dtype=np.float32)
    fp16_array = fp32_array.astype(np.float16)
    fp16_bytes = fp16_array.tobytes()
    print(f"  fp32: {len(fp32_data) / (1024**2):.1f} MB -> fp16: {len(fp16_bytes) / (1024**2):.1f} MB")

    # Update initializer data type
    embed_init.data_type = TensorProto.FLOAT16

    # Find the Gather node that uses this embedding
    gather_node = None
    for node in model.graph.node:
        if node.op_type == "Gather" and EMBED_TENSOR_NAME in node.input:
            gather_node = node
            break
    assert gather_node is not None, "Could not find Gather node using embed_tokens"

    # Rename Gather output and insert Cast node
    original_output = gather_node.output[0]  # "embedding"
    fp16_output = original_output + "_fp16"
    gather_node.output[0] = fp16_output

    cast_node = helper.make_node(
        "Cast",
        inputs=[fp16_output],
        outputs=[original_output],
        name="cast_embedding_fp16_to_fp32",
        to=TensorProto.FLOAT,  # Cast to fp32
    )

    # Insert Cast node right after Gather
    gather_idx = list(model.graph.node).index(gather_node)
    model.graph.node.insert(gather_idx + 1, cast_node)
    print(f"Inserted Cast node: {fp16_output} -> {original_output}")

    return fp16_bytes, model


def get_external_data_info(init):
    """Extract external data info dict from an initializer."""
    return {e.key: e.value for e in init.external_data}


def set_external_data(init, location, offset, length):
    """Update external data references on an initializer."""
    # Clear existing external_data and rebuild
    while len(init.external_data) > 0:
        init.external_data.pop()
    init.external_data.add(key="location", value=location)
    init.external_data.add(key="offset", value=str(offset))
    init.external_data.add(key="length", value=str(length))


def reshard_all_data(model, embed_fp16_bytes):
    """
    Re-shard all external data into files < MAX_SHARD_SIZE.
    The embed_tokens data is replaced with fp16_bytes.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all tensors with external data
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

    print(f"\nRe-sharding {len(tensors)} tensors...")

    # Sort by (location, offset)
    tensors.sort(key=lambda t: (t["location"], t["offset"]))

    # Open source files for reading non-embed tensors
    src_files = {}
    for t in tensors:
        loc = t["location"]
        if loc not in src_files:
            src_files[loc] = open(INPUT_DIR / loc, "rb")

    # Greedy bin-packing
    base_name = DECODER_ONNX
    shard_idx = 0
    shard_offset = 0
    shard_file = None
    shard_path = None

    def open_new_shard():
        nonlocal shard_idx, shard_offset, shard_file, shard_path
        if shard_file is not None:
            size_mb = shard_offset / (1024 ** 2)
            print(f"  Closed shard {shard_path.name}: {size_mb:.1f} MB")
            shard_file.close()
            shard_idx += 1
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        shard_name = f"{base_name}_data{suffix}"
        shard_path = OUTPUT_DIR / shard_name
        shard_file = open(shard_path, "wb")
        shard_offset = 0

    open_new_shard()

    for t in tensors:
        is_embed = t["init"].name == EMBED_TENSOR_NAME

        if is_embed:
            data = embed_fp16_bytes
            data_len = len(data)
        else:
            data_len = t["length"]

        # Check if we need a new shard
        if shard_offset + data_len > MAX_SHARD_SIZE and shard_offset > 0:
            open_new_shard()

        if not is_embed:
            src_f = src_files[t["location"]]
            src_f.seek(t["offset"])
            data = src_f.read(data_len)
            assert len(data) == data_len

        # Write to shard
        new_offset = shard_offset
        shard_file.write(data)
        shard_offset += data_len

        # Update protobuf
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        current_shard_name = f"{base_name}_data{suffix}"
        set_external_data(t["init"], current_shard_name, new_offset, data_len)

    # Close last shard
    size_mb = shard_offset / (1024 ** 2)
    print(f"  Closed shard {shard_path.name}: {size_mb:.1f} MB")
    shard_file.close()
    num_shards = shard_idx + 1

    # Close source files
    for f in src_files.values():
        f.close()

    print(f"\nCreated {num_shards} shards (all < {MAX_SHARD_SIZE / (1024**3):.1f} GB)")
    return num_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    print(f"Loading {DECODER_ONNX}...")
    model = onnx.load(str(INPUT_DIR / DECODER_ONNX), load_external_data=False)

    # Step 1: Convert embedding to fp16
    embed_fp16_bytes, model = convert_embedding_to_fp16(model)

    # Step 2: Re-shard all external data
    num_shards = reshard_all_data(model, embed_fp16_bytes)

    # Step 3: Save updated ONNX model
    output_onnx = OUTPUT_DIR / DECODER_ONNX
    onnx.save(model, str(output_onnx))
    print(f"Saved {output_onnx.name}")

    # Copy encoder files (unchanged)
    import shutil
    for name in ["encoder_model_q4.onnx", "encoder_model_q4.onnx_data"]:
        src = INPUT_DIR / name
        dst = OUTPUT_DIR / name
        if src.exists():
            if dst.exists():
                dst.unlink()
            print(f"Copying {name}")
            shutil.copy2(src, dst)

    # Summary
    print(f"\nAll files in {OUTPUT_DIR}:")
    total = 0
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        total += size_mb
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"  Total: {total / 1024:.2f} GB")
    print(f"\n*** Update worker.js: decoder_model_merged: {num_shards} ***")

    if args.upload:
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
    else:
        print("\nTo upload, run again with --upload flag.")


if __name__ == "__main__":
    main()
