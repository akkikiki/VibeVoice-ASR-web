#!/usr/bin/env python3
"""
Streaming INT8 quantization of the FP32 decoder for VibeVoice-ASR.

Unlike onnxruntime's quantize_dynamic (which loads the full 29 GB model into RAM),
this script streams tensor data from disk one-at-a-time, keeping memory usage low.

Strategy (mirrors convert_encoder_fp16_v2.py):
  - Load only the ONNX protobuf (~few MB), NOT the external data
  - For each FP32 weight used by MatMul nodes:
      Read from disk → quantize to INT8 → write INT8 data to output
      Add scale/zero_point initializers + DequantizeLinear node
  - For embed_tokens: FP32 → FP16 + Cast node (same as before)
  - For everything else (biases, layer norms): copy as-is
  - Reshard output into files < 1.9 GB for browser compatibility

Quantization details:
  - Symmetric per-tensor: scale = max(|w|) / 127, zero_point = 0
  - INT8 weight = round(w / scale).clip(-128, 127)
  - DequantizeLinear at runtime: FP32 = (INT8 - 0) * scale

Requirements:
    pip install onnx numpy huggingface_hub

Usage:
    python quantize_decoder_int8.py                   # local only
    python quantize_decoder_int8.py --upload           # quantize + upload
    python quantize_decoder_int8.py --skip-download    # reuse cached files
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
WORK_DIR = Path("/tmp/vibevoice-split/decoder_int8")
INPUT_DIR = WORK_DIR / "onnx"
OUTPUT_DIR = WORK_DIR / "resharded"
MAX_SHARD_SIZE = 1_900_000_000  # 1.9 GB

SRC_ONNX = "decoder_with_speech.onnx"
DST_ONNX = "decoder_model_merged_q8.onnx"
EMBED_TENSOR_NAME = "language_model.embed_tokens.weight"


def download_decoder():
    """Download the FP32 decoder from HuggingFace."""
    from huggingface_hub import hf_hub_download

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading decoder_with_speech.onnx...")
    hf_hub_download(REPO_ID, "onnx/decoder_with_speech.onnx", local_dir=WORK_DIR)
    print("Downloading decoder_with_speech.onnx.data (29 GB)...")
    hf_hub_download(REPO_ID, "onnx/decoder_with_speech.onnx.data", local_dir=WORK_DIR)
    print("Done!")


def clear_external_data(init):
    """Remove all external_data entries from an initializer."""
    while len(init.external_data) > 0:
        init.external_data.pop()


def set_external_data(init, location, offset, length):
    """Set external_data entries on an initializer."""
    clear_external_data(init)
    init.external_data.add(key="location", value=location)
    init.external_data.add(key="offset", value=str(offset))
    init.external_data.add(key="length", value=str(length))


def get_external_info(init):
    """Get external data info dict from an initializer."""
    return {e.key: e.value for e in init.external_data}


def read_tensor_from_disk(src_dir, info):
    """Read raw tensor bytes from an external data file."""
    src_file = src_dir / info["location"]
    src_offset = int(info.get("offset", "0"))
    src_length = int(info["length"])
    with open(src_file, "rb") as f:
        f.seek(src_offset)
        data = f.read(src_length)
    assert len(data) == src_length, f"Short read: {len(data)} vs {src_length}"
    return data


def quantize_tensor_int8(fp32_bytes):
    """
    Symmetric per-tensor INT8 quantization.

    Returns:
        int8_bytes: quantized weight data
        scale: FP32 scalar (max(|w|) / 127)
    """
    arr = np.frombuffer(fp32_bytes, dtype=np.float32)
    abs_max = np.max(np.abs(arr))
    if abs_max == 0:
        scale = np.float32(1.0)
    else:
        scale = np.float32(abs_max / 127.0)
    quantized = np.round(arr / scale).clip(-128, 127).astype(np.int8)
    return quantized.tobytes(), scale


def quantize_streaming():
    """
    Stream through all initializers, quantize MatMul weights to INT8,
    convert embed_tokens to FP16, copy everything else, and reshard.
    """
    print("\n=== Loading protobuf (no external data) ===")
    model = onnx.load(str(INPUT_DIR / SRC_ONNX), load_external_data=False)
    src_dir = INPUT_DIR

    # --- Identify which initializers are used by MatMul nodes ---
    matmul_weight_names = set()
    for node in model.graph.node:
        if node.op_type == "MatMul":
            # MatMul has 2 inputs; typically input[1] is the weight initializer
            for inp_name in node.input:
                matmul_weight_names.add(inp_name)

    # Filter to only those that are actually initializers (not activations)
    init_names = {init.name for init in model.graph.initializer}
    matmul_weight_names = matmul_weight_names & init_names

    # --- Collect FP32 initializers with external data ---
    fp32_external_inits = {}
    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" not in info:
            continue
        if init.data_type == TensorProto.FLOAT:
            fp32_external_inits[init.name] = init

    # Classify which FP32 weights to quantize vs FP16 vs skip
    quantize_names = matmul_weight_names & set(fp32_external_inits.keys())
    embed_name = EMBED_TENSOR_NAME if EMBED_TENSOR_NAME in fp32_external_inits else None

    # Don't INT8-quantize the embedding — it gets FP16
    quantize_names.discard(EMBED_TENSOR_NAME)

    print(f"Total initializers: {len(list(model.graph.initializer))}")
    print(f"FP32 external initializers: {len(fp32_external_inits)}")
    print(f"MatMul weights to INT8 quantize: {len(quantize_names)}")
    if embed_name:
        print(f"Embedding to FP16: {embed_name}")

    # --- Step 1: Modify protobuf for quantized weights ---
    # For each quantized weight:
    #   - Rename initializer: {name} -> {name}_quantized (INT8)
    #   - Add scale initializer: {name}_scale (FP32 scalar)
    #   - Add zero_point initializer: {name}_zero_point (INT8 scalar)
    #   - Insert DequantizeLinear node: quantized + scale + zp -> {name}
    print("\n=== Modifying protobuf for INT8 quantization ===")

    dequant_nodes = []
    scale_inits_to_add = []  # (name, will be written inline in protobuf)
    zp_inits_to_add = []

    for name in sorted(quantize_names):
        init = fp32_external_inits[name]
        q_name = name + "_quantized"
        scale_name = name + "_scale"
        zp_name = name + "_zero_point"

        # Rename initializer to quantized version
        init.name = q_name
        init.data_type = TensorProto.INT8

        # DequantizeLinear node: q_name + scale + zp -> original name
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, scale_name, zp_name],
            outputs=[name],
            name=f"dequant_{name.replace('.', '_')}",
        )
        dequant_nodes.append(dq_node)

    # --- Step 2: Handle embed_tokens -> FP16 + Cast ---
    if embed_name:
        embed_init = fp32_external_inits[embed_name]
        fp16_name = embed_name + "_fp16"
        embed_init.name = fp16_name
        embed_init.data_type = TensorProto.FLOAT16

        # Find Gather node that uses embed_tokens
        gather_node = None
        for node in model.graph.node:
            if node.op_type == "Gather" and embed_name in node.input:
                gather_node = node
                break

        if gather_node:
            # Update Gather input to use FP16 initializer name
            for idx, inp in enumerate(gather_node.input):
                if inp == embed_name:
                    gather_node.input[idx] = fp16_name

            original_output = gather_node.output[0]
            fp16_output = original_output + "_fp16"
            gather_node.output[0] = fp16_output

            cast_node = helper.make_node(
                "Cast",
                inputs=[fp16_output],
                outputs=[original_output],
                name="cast_embedding_fp16_to_fp32",
                to=TensorProto.FLOAT,
            )

            gather_idx = list(model.graph.node).index(gather_node)
            model.graph.node.insert(gather_idx + 1, cast_node)
            print(f"  Inserted Cast node for embed_tokens: {fp16_output} -> {original_output}")

    # Insert all DequantizeLinear nodes at the beginning of the graph
    for i, dq_node in enumerate(dequant_nodes):
        model.graph.node.insert(i, dq_node)
    print(f"  Inserted {len(dequant_nodes)} DequantizeLinear nodes")

    # --- Step 3: Stream data, quantize, and write to sharded output ---
    print("\n=== Streaming quantization + resharding ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean output directory
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    base_name = DST_ONNX
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

    def write_to_shard(data):
        """Write data to the current shard, opening a new one if needed."""
        nonlocal shard_offset
        data_len = len(data)
        if shard_offset + data_len > MAX_SHARD_SIZE and shard_offset > 0:
            open_new_shard()
        new_offset = shard_offset
        shard_file.write(data)
        shard_offset += data_len
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        return f"{base_name}_data{suffix}", new_offset, data_len

    open_new_shard()

    total_fp32_size = 0
    total_int8_size = 0
    total_fp16_size = 0
    quantized_count = 0

    # Process all initializers in order
    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" not in info:
            # Inline initializer (no external data) — skip, it stays in protobuf
            continue

        # Determine the original name before our renaming
        original_name = init.name
        is_quantized = original_name.endswith("_quantized")
        is_fp16_embed = original_name.endswith("_fp16") and original_name.removesuffix("_fp16") == EMBED_TENSOR_NAME

        # Read raw FP32 data from disk
        raw_data = read_tensor_from_disk(src_dir, info)

        if is_quantized:
            # Quantize FP32 -> INT8
            int8_bytes, scale_val = quantize_tensor_int8(raw_data)
            total_fp32_size += len(raw_data)
            total_int8_size += len(int8_bytes)
            quantized_count += 1

            if quantized_count % 50 == 0:
                print(f"  Quantized {quantized_count} tensors...")

            # Write INT8 weight data to shard
            loc, off, length = write_to_shard(int8_bytes)
            set_external_data(init, loc, off, length)

            # Create scale initializer (inline in protobuf — it's just 4 bytes)
            base_weight_name = original_name.removesuffix("_quantized")
            scale_init = numpy_helper.from_array(
                np.array(scale_val, dtype=np.float32),
                name=base_weight_name + "_scale",
            )
            model.graph.initializer.append(scale_init)

            # Create zero_point initializer (inline — 1 byte)
            zp_init = numpy_helper.from_array(
                np.array(0, dtype=np.int8),
                name=base_weight_name + "_zero_point",
            )
            model.graph.initializer.append(zp_init)

        elif is_fp16_embed:
            # Convert FP32 -> FP16
            arr = np.frombuffer(raw_data, dtype=np.float32)
            fp16_bytes = arr.astype(np.float16).tobytes()
            total_fp32_size += len(raw_data)
            total_fp16_size += len(fp16_bytes)
            print(f"  embed_tokens FP32: {len(raw_data) / (1024**2):.1f} MB -> FP16: {len(fp16_bytes) / (1024**2):.1f} MB")

            loc, off, length = write_to_shard(fp16_bytes)
            set_external_data(init, loc, off, length)

        else:
            # Copy as-is (biases, layer norms, non-FP32, etc.)
            loc, off, length = write_to_shard(raw_data)
            set_external_data(init, loc, off, length)

    # Close last shard
    if shard_file is not None:
        size_mb = shard_offset / (1024 ** 2)
        print(f"  Closed shard {shard_path.name}: {size_mb:.1f} MB")
        shard_file.close()
    num_shards = shard_idx + 1

    print(f"\n  Quantized {quantized_count} MatMul weights to INT8")
    print(f"  FP32 -> INT8: {total_fp32_size / (1024**2):.1f} MB -> {total_int8_size / (1024**2):.1f} MB (weights)")
    if total_fp16_size > 0:
        print(f"  FP32 -> FP16: embed_tokens ({total_fp16_size / (1024**2):.1f} MB)")
    print(f"  Created {num_shards} shards")

    # Save ONNX protobuf
    out_onnx = OUTPUT_DIR / DST_ONNX
    onnx.save(model, str(out_onnx))
    print(f"  Saved {out_onnx.name}")

    # Print summary
    print(f"\nAll files in {OUTPUT_DIR}:")
    total = 0
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        total += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  Total: {total / 1024:.2f} GB")

    return OUTPUT_DIR, num_shards


def upload(output_dir: Path):
    """Upload to HuggingFace."""
    print("\n=== Upload ===")
    from huggingface_hub import HfApi
    api = HfApi()

    for f in sorted(output_dir.iterdir()):
        remote_path = f"onnx/{f.name}"
        print(f"Uploading {f.name} ({f.stat().st_size / (1024**2):.1f} MB) -> {remote_path}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
        )
    print("Upload complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming INT8 quantization of VibeVoice-ASR decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--upload", action="store_true", help="Upload results to HuggingFace")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading (use cached files)")
    args = parser.parse_args()

    if not args.skip_download:
        download_decoder()

    output_dir, num_shards = quantize_streaming()

    print(f"\n*** Update worker.js: decoder_model_merged dtype to 'q8', shards to {num_shards} ***")
    print(f"*** Update HF config.json: decoder_model_merged dtype to 'q8', shards to {num_shards} ***")

    if args.upload:
        upload(output_dir)
    else:
        print("\nTo upload, run again with --upload flag.")


if __name__ == "__main__":
    main()
