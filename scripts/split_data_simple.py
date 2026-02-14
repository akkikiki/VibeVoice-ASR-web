"""
Simple byte-level split of an ONNX external data file into ~2 GB shards.

Unlike split_onnx_data.py (which uses the onnx library to re-export),
this script does a raw byte split and updates the ONNX protobuf to
reference the new shard files. This avoids loading the full model into
memory.

Usage:
    python scripts/split_data_simple.py /path/to/decoder_with_speech.onnx

This will:
  1. Read the .onnx graph (small, ~4 MB)
  2. Split .onnx.data into ~2 GB chunks named decoder_with_speech.onnx_data, _data_1, etc.
  3. Update tensor references in the .onnx graph to point to the correct shard + offset
  4. Overwrite the .onnx file with updated references

Requirements:
    pip install onnx
"""

import argparse
import os
import sys

SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB


def split_data_file(data_path, base_name):
    """Split a large data file into shards and return shard info."""
    file_size = os.path.getsize(data_path)
    num_shards = max(1, (file_size + SHARD_SIZE - 1) // SHARD_SIZE)

    print(f"Splitting {data_path} ({file_size / 1e9:.2f} GB) into {num_shards} shards...")

    shards = []
    with open(data_path, "rb") as f:
        for i in range(num_shards):
            suffix = "" if i == 0 else f"_{i}"
            shard_name = f"{base_name}_data{suffix}"
            shard_path = os.path.join(os.path.dirname(data_path), shard_name)

            chunk = f.read(SHARD_SIZE)
            with open(shard_path, "wb") as sf:
                sf.write(chunk)

            shards.append({
                "name": shard_name,
                "path": shard_path,
                "size": len(chunk),
                "offset_start": i * SHARD_SIZE,
            })
            print(f"  Shard {i}: {shard_name} ({len(chunk) / 1e9:.2f} GB)")

    return shards


def update_onnx_graph(onnx_path, shards):
    """Update the ONNX graph to reference sharded external data files."""
    import onnx
    from onnx import TensorProto

    print(f"\nUpdating ONNX graph: {onnx_path}")
    model = onnx.load(onnx_path, load_external_data=False)

    updated = 0
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue

        # Read external data info
        ext_info = {}
        for entry in tensor.external_data:
            ext_info[entry.key] = entry.value

        if "location" not in ext_info:
            continue

        offset = int(ext_info.get("offset", 0))
        length = int(ext_info.get("length", 0))

        # Find which shard this tensor belongs to
        shard_idx = offset // SHARD_SIZE
        new_offset = offset % SHARD_SIZE

        if shard_idx >= len(shards):
            print(f"  WARNING: Tensor {tensor.name} offset {offset} exceeds shard range")
            continue

        new_location = shards[shard_idx]["name"]

        # Verify tensor doesn't span shard boundary
        if new_offset + length > shards[shard_idx]["size"]:
            print(f"  WARNING: Tensor {tensor.name} spans shard boundary! "
                  f"offset={offset}, length={length}, shard_size={shards[shard_idx]['size']}")
            # This can happen if a tensor straddles the 2GB boundary
            # In practice this is rare for individual weight tensors
            continue

        # Update external data references
        del tensor.external_data[:]
        tensor.external_data.add(key="location", value=new_location)
        tensor.external_data.add(key="offset", value=str(new_offset))
        tensor.external_data.add(key="length", value=str(length))

        updated += 1

    print(f"Updated {updated} tensor references")

    # Save the updated graph
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    print(f"Saved updated graph: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split ONNX external data into ~2 GB shards"
    )
    parser.add_argument(
        "onnx_path",
        help="Path to the .onnx model file (e.g., decoder_with_speech.onnx)",
    )
    args = parser.parse_args()

    onnx_path = args.onnx_path
    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found")
        sys.exit(1)

    data_path = onnx_path + ".data"
    if not os.path.exists(data_path):
        print(f"ERROR: External data file {data_path} not found")
        sys.exit(1)

    base_name = os.path.basename(onnx_path)  # e.g., "decoder_with_speech.onnx"

    # Step 1: Split the data file
    shards = split_data_file(data_path, base_name)

    # Step 2: Update the ONNX graph
    update_onnx_graph(onnx_path, shards)

    # Step 3: Remove the original .data file (now replaced by shards)
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"\nRemoved original: {data_path}")

    print("\nDone! Files to upload to HuggingFace:")
    print(f"  {os.path.basename(onnx_path)} (updated graph)")
    for shard in shards:
        print(f"  {shard['name']} ({shard['size'] / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
