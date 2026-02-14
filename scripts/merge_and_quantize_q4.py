#!/usr/bin/env python3
"""
Merge split KV-cache decoders into decoder_model_merged, then quantize to Q4.

Steps:
  1. Download FP32 KV-cache exports from HuggingFace (onnx_kvcache/)
  2. Merge decoder_model.onnx + decoder_with_past_model.onnx using optimum
  3. Quantize to Q4 (4-bit block quantization via MatMulNBits)
  4. Reshard external data to < 1.9 GB for browser compatibility
  5. Optionally upload to HuggingFace

Requirements:
    pip install onnx onnxruntime optimum-onnx huggingface_hub

Usage:
    python scripts/merge_and_quantize_q4.py                    # merge + quantize
    python scripts/merge_and_quantize_q4.py --upload            # + upload to HF
    python scripts/merge_and_quantize_q4.py --skip-download     # reuse cached files
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
WORK_DIR = Path("/tmp/vibevoice-split/decoder_q4_kvcache")
INPUT_DIR = Path("/tmp/vibevoice-split/decoder_kvcache_int8/onnx_kvcache")
MERGED_DIR = WORK_DIR / "merged"
OUTPUT_DIR = WORK_DIR / "q4_resharded"
MAX_SHARD_SIZE = 1_900_000_000  # 1.9 GB

EMBED_TENSOR_NAME = "language_model.embed_tokens.weight"
BLOCK_SIZE = 32  # Standard block size for Q4 quantization


def download_kvcache():
    """Download KV-cache decoder files from HuggingFace."""
    from huggingface_hub import HfApi, hf_hub_download

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    all_files = api.list_repo_files(REPO_ID)
    kvcache_files = [f for f in all_files if f.startswith("onnx_kvcache/")]

    print(f"Downloading {len(kvcache_files)} files from onnx_kvcache/...")
    for i, f in enumerate(kvcache_files):
        if i % 50 == 0:
            print(f"  {i}/{len(kvcache_files)}...")
        hf_hub_download(REPO_ID, f, local_dir=INPUT_DIR.parent)
    print("Done!")


def merge_decoders_step():
    """Merge prefill + decode-with-past into decoder_model_merged."""
    from optimum.onnx import merge_decoders

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    prefill_path = INPUT_DIR / "decoder_model.onnx"
    decode_path = INPUT_DIR / "decoder_with_past_model.onnx"

    print(f"\n{'='*60}")
    print("Step 1: Merge decoders")
    print(f"  Prefill: {prefill_path}")
    print(f"  Decode:  {decode_path}")
    print(f"{'='*60}")

    # Load both models
    print("Loading prefill model...")
    prefill = onnx.load(str(prefill_path), load_external_data=True)
    print(f"  Prefill nodes: {len(prefill.graph.node)}")

    print("Loading decode-with-past model...")
    decode = onnx.load(str(decode_path), load_external_data=True)
    print(f"  Decode nodes: {len(decode.graph.node)}")

    # Merge using optimum
    print("Merging models...")
    merged_path = MERGED_DIR / "decoder_model_merged.onnx"
    merged = merge_decoders(prefill, decode, save_path=None, strict=False)
    print(f"  Merged nodes: {len(merged.graph.node)}")

    # Save with external data
    print(f"Saving merged model to {merged_path}...")
    onnx.save(
        merged,
        str(merged_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder_model_merged.onnx_data",
    )

    # Print merged model I/O
    print("\nMerged model inputs:")
    for inp in merged.graph.input[:5]:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print(f"  ... ({len(merged.graph.input)} total)")
    print("Merged model outputs:")
    for out in merged.graph.output[:3]:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    print(f"  ... ({len(merged.graph.output)} total)")

    del prefill, decode, merged
    return merged_path


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


def quantize_tensor_q4(fp32_bytes, block_size=BLOCK_SIZE):
    """
    Symmetric Q4 block quantization (INT4, packed as uint8).

    For each block of `block_size` FP32 values:
      - scale = max(|block|) / 7.0
      - quantized = round(block / scale), clamped to [-8, 7]
      - pack two INT4 values per byte (low nibble first)

    Returns:
        packed_bytes: uint8 array with packed INT4 values
        scales: FP32 array of per-block scales
        zero_points: uint8 array (all zeros for symmetric)
    """
    arr = np.frombuffer(fp32_bytes, dtype=np.float32)
    n = len(arr)

    # Pad to multiple of block_size
    if n % block_size != 0:
        padded = np.zeros(((n + block_size - 1) // block_size) * block_size, dtype=np.float32)
        padded[:n] = arr
        arr = padded

    num_blocks = len(arr) // block_size
    arr_blocks = arr.reshape(num_blocks, block_size)

    # Compute per-block scales
    abs_max = np.max(np.abs(arr_blocks), axis=1)
    scales = (abs_max / 7.0).astype(np.float32)
    scales[scales == 0] = 1.0  # avoid div by zero

    # Quantize to INT4 range [-8, 7]
    quantized = np.round(arr_blocks / scales[:, None]).clip(-8, 7).astype(np.int8)

    # Offset to unsigned [0, 15] for packing
    quantized_unsigned = (quantized + 8).astype(np.uint8)

    # Pack two values per byte (low nibble first)
    assert block_size % 2 == 0
    packed = np.zeros((num_blocks, block_size // 2), dtype=np.uint8)
    for i in range(block_size // 2):
        packed[:, i] = quantized_unsigned[:, 2 * i] | (quantized_unsigned[:, 2 * i + 1] << 4)

    # Zero points (all 8 for symmetric around 0, packed as pairs)
    zero_points = np.full((num_blocks, (block_size + 1) // 2 // 4 + (1 if block_size <= 32 else 0)),
                          0x88, dtype=np.uint8)
    # Actually for MatMulNBits, zero_point is simpler
    zero_points = np.full(num_blocks, 8, dtype=np.uint8)

    return packed.tobytes(), scales, zero_points


def quantize_to_q4(merged_path):
    """
    Quantize merged decoder to Q4 using MatMulNBits nodes.

    For ONNX Runtime WebGPU, Q4 quantization uses the MatMulNBits op:
      - Input: activation (FP32)
      - B: packed INT4 weights (uint8)
      - scales: FP32 per-block scales
      - zero_points: uint8 per-block zero points

    This replaces MatMul(A, W) with MatMulNBits(A, B_q4, scales, zp).
    """
    print(f"\n{'='*60}")
    print("Step 2: Quantize to Q4")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged model...")
    model = onnx.load(str(merged_path), load_external_data=False)
    src_dir = merged_path.parent

    # Find MatMul weight initializers
    matmul_weight_names = set()
    matmul_nodes = []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            matmul_nodes.append(node)
            for inp_name in node.input:
                matmul_weight_names.add(inp_name)

    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init

    # Filter to only actual weight initializers (not activations)
    matmul_weight_names = matmul_weight_names & set(init_map.keys())
    matmul_weight_names.discard(EMBED_TENSOR_NAME)

    # Only quantize FP32 external tensors
    fp32_weights = {}
    for name in matmul_weight_names:
        init = init_map[name]
        info = get_external_info(init)
        if "location" in info and init.data_type == TensorProto.FLOAT:
            fp32_weights[name] = init

    print(f"  MatMul FP32 weights to quantize: {len(fp32_weights)}")

    # Read source data file
    src_data_info = {}
    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" in info:
            src_data_info[init.name] = info

    # Process each MatMul node
    nodes_to_remove = []
    nodes_to_add = []
    inits_to_add = []
    inits_to_remove = set()
    quantized_count = 0

    for node in matmul_nodes:
        # Find which input is the weight
        weight_input = None
        weight_idx = None
        for idx, inp_name in enumerate(node.input):
            if inp_name in fp32_weights:
                weight_input = inp_name
                weight_idx = idx
                break

        if weight_input is None:
            continue

        init = fp32_weights[weight_input]
        info = src_data_info.get(weight_input)
        if info is None:
            continue

        # Read weight data
        src_file = src_dir / info["location"]
        offset = int(info.get("offset", "0"))
        length = int(info["length"])
        with open(src_file, "rb") as f:
            f.seek(offset)
            raw_data = f.read(length)
        assert len(raw_data) == length

        # Get weight shape
        dims = list(init.dims)
        if len(dims) != 2:
            continue  # Only handle 2D weights

        K, N = dims  # weight shape for MatMul: [K, N] where output = input @ weight

        # Quantize
        packed_bytes, scales, zero_points = quantize_tensor_q4(raw_data, BLOCK_SIZE)

        quantized_count += 1
        if quantized_count % 20 == 0:
            print(f"  Quantized {quantized_count} weights...")

        # Create MatMulNBits node
        b_name = f"{weight_input}_q4"
        scales_name = f"{weight_input}_scales"
        zp_name = f"{weight_input}_zp"

        # MatMulNBits expects B in shape [N, (K + block_size - 1) // block_size, block_size // 2]
        # But the actual layout is: B is packed as [N, ceil(K/block_size), block_size/2]
        # For our case: input @ weight where weight is [K, N]
        # MatMulNBits: Y = A @ dequant(B) where B represents [K, N] in packed form
        # B shape: [N, n_blocks_per_col, blob_size] where n_blocks = ceil(K/bs), blob_size = bs/2
        n_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        blob_size = BLOCK_SIZE // 2

        # Reshape packed data: currently [K*N/2] bytes -> [N, n_blocks, blob_size]
        # But our quantization was done row-wise on [K, N] flattened...
        # MatMulNBits expects column-wise packing: for each output channel n, pack K values
        # We need to requantize column-wise

        # Re-read as float array and quantize column-wise
        weight_arr = np.frombuffer(raw_data, dtype=np.float32).reshape(K, N)

        # Transpose to [N, K] for column-wise quantization
        weight_t = weight_arr.T.copy()  # [N, K]

        # Pad K to multiple of block_size
        K_padded = ((K + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if K_padded != K:
            weight_padded = np.zeros((N, K_padded), dtype=np.float32)
            weight_padded[:, :K] = weight_t
            weight_t = weight_padded

        n_blocks_k = K_padded // BLOCK_SIZE
        weight_blocks = weight_t.reshape(N, n_blocks_k, BLOCK_SIZE)

        # Per-block scales
        block_abs_max = np.max(np.abs(weight_blocks), axis=2)
        block_scales = (block_abs_max / 7.0).astype(np.float32)
        block_scales[block_scales == 0] = 1.0

        # Quantize to INT4 [-8, 7]
        quantized_blocks = np.round(weight_blocks / block_scales[:, :, None]).clip(-8, 7).astype(np.int8)

        # Offset to unsigned [0, 15]
        q_unsigned = (quantized_blocks + 8).astype(np.uint8)

        # Pack two values per byte
        packed = np.zeros((N, n_blocks_k, BLOCK_SIZE // 2), dtype=np.uint8)
        for bi in range(BLOCK_SIZE // 2):
            packed[:, :, bi] = q_unsigned[:, :, 2 * bi] | (q_unsigned[:, :, 2 * bi + 1] << 4)

        # Zero points: packed, 8 for symmetric
        # For MatMulNBits: zp shape is [N, ceil(n_blocks_k / 2)] packed uint8
        zp_packed_cols = (n_blocks_k + 1) // 2
        zp_packed = np.full((N, zp_packed_cols), 0x88, dtype=np.uint8)

        # Create initializer tensors (inline, small enough)
        b_init = numpy_helper.from_array(packed, name=b_name)
        scales_init = numpy_helper.from_array(block_scales, name=scales_name)
        zp_init = numpy_helper.from_array(zp_packed, name=zp_name)

        inits_to_add.extend([b_init, scales_init, zp_init])
        inits_to_remove.add(weight_input)

        # Replace MatMul with MatMulNBits
        activation_input = node.input[1 - weight_idx]  # the other input

        matmul_nbits = helper.make_node(
            "MatMulNBits",
            inputs=[activation_input, b_name, scales_name, zp_name],
            outputs=node.output,
            name=node.name + "_q4" if node.name else f"matmulnbits_{quantized_count}",
            domain="com.microsoft",
            K=K,
            N=N,
            bits=4,
            block_size=BLOCK_SIZE,
        )

        nodes_to_remove.append(node)
        nodes_to_add.append(matmul_nbits)

    print(f"  Total quantized: {quantized_count} MatMul -> MatMulNBits")

    # Apply changes to the graph
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    model.graph.node.extend(nodes_to_add)

    # Remove old initializers and add new ones
    new_inits = []
    for init in model.graph.initializer:
        if init.name in inits_to_remove:
            continue
        new_inits.append(init)
    new_inits.extend(inits_to_add)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Handle embed_tokens -> FP16
    embed_init = init_map.get(EMBED_TENSOR_NAME)
    if embed_init and get_external_info(embed_init).get("location"):
        print("  Converting embed_tokens to FP16...")
        info = src_data_info[EMBED_TENSOR_NAME]
        src_file = src_dir / info["location"]
        with open(src_file, "rb") as f:
            f.seek(int(info.get("offset", "0")))
            raw = f.read(int(info["length"]))

        fp16_arr = np.frombuffer(raw, dtype=np.float32).astype(np.float16)

        # Find embed init in new_inits and update
        for init in model.graph.initializer:
            if init.name == EMBED_TENSOR_NAME:
                fp16_name = EMBED_TENSOR_NAME + "_fp16"
                init.name = fp16_name
                init.data_type = TensorProto.FLOAT16
                clear_external_data(init)
                init.raw_data = fp16_arr.tobytes()

                # Update Gather node
                for node in model.graph.node:
                    if node.op_type == "Gather":
                        for idx, inp in enumerate(node.input):
                            if inp == EMBED_TENSOR_NAME:
                                node.input[idx] = fp16_name
                                original_out = node.output[0]
                                fp16_out = original_out + "_fp16"
                                node.output[0] = fp16_out
                                # Add Cast node
                                cast = helper.make_node(
                                    "Cast",
                                    inputs=[fp16_out],
                                    outputs=[original_out],
                                    name="cast_embed_fp16_to_fp32",
                                    to=TensorProto.FLOAT,
                                )
                                node_idx = list(model.graph.node).index(node)
                                model.graph.node.insert(node_idx + 1, cast)
                                break
                break

    # Handle remaining external data (non-quantized tensors)
    # Copy them to inline or to output shards
    print("\n  Resharding remaining external data...")

    out_name = "decoder_model_merged_q4.onnx"
    shard_idx = 0
    shard_offset = 0
    shard_file = None
    shard_path = None

    def open_new_shard():
        nonlocal shard_idx, shard_offset, shard_file, shard_path
        if shard_file is not None:
            size_mb = shard_offset / (1024 ** 2)
            print(f"    Closed shard {shard_path.name}: {size_mb:.1f} MB")
            shard_file.close()
            shard_idx += 1
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        shard_name = f"{out_name}_data{suffix}"
        shard_path = OUTPUT_DIR / shard_name
        shard_file = open(shard_path, "wb")
        shard_offset = 0

    def write_to_shard(data):
        nonlocal shard_offset
        data_len = len(data)
        if shard_offset + data_len > MAX_SHARD_SIZE and shard_offset > 0:
            open_new_shard()
        new_offset = shard_offset
        shard_file.write(data)
        shard_offset += data_len
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        return f"{out_name}_data{suffix}", new_offset, data_len

    open_new_shard()

    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" not in info:
            # Inline data (Q4 packed weights, scales, zp, etc.)
            if init.raw_data and len(init.raw_data) > 1024:
                # Large inline -> write to shard
                loc, off, length = write_to_shard(init.raw_data)
                init.raw_data = b""
                init.data_location = TensorProto.EXTERNAL
                set_external_data(init, loc, off, length)
            continue

        # External data that wasn't quantized — copy to output shard
        src_file = src_dir / info["location"]
        src_offset = int(info.get("offset", "0"))
        src_length = int(info["length"])

        with open(src_file, "rb") as f:
            f.seek(src_offset)
            data = f.read(src_length)
        assert len(data) == src_length

        loc, off, length = write_to_shard(data)
        set_external_data(init, loc, off, length)

    if shard_file is not None:
        size_mb = shard_offset / (1024 ** 2)
        print(f"    Closed shard {shard_path.name}: {size_mb:.1f} MB")
        shard_file.close()
    num_shards = shard_idx + 1

    # Save ONNX protobuf
    out_onnx_path = OUTPUT_DIR / out_name
    onnx.save(model, str(out_onnx_path))
    print(f"\n  Saved {out_name}")

    # Print summary
    print(f"\nOutput files in {OUTPUT_DIR}:")
    total = 0
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        total += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  Total: {total / 1024:.2f} GB")

    return num_shards


def upload_files():
    """Upload Q4 merged decoder files to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi()

    print(f"\n{'='*60}")
    print("Uploading to HuggingFace")
    print(f"{'='*60}")

    for f in sorted(OUTPUT_DIR.iterdir()):
        remote_path = f"onnx/{f.name}"
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"Uploading {f.name} ({size_mb:.1f} MB) -> {remote_path}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
        )
    print("Upload complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and quantize KV-cache decoders to Q4",
    )
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (reuse existing)")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if INPUT_DIR.exists() and (INPUT_DIR / "decoder_model.onnx").exists():
            print("KV-cache files already downloaded, skipping download.")
        else:
            download_kvcache()

    if args.skip_merge and (MERGED_DIR / "decoder_model_merged.onnx").exists():
        print("Using existing merged model.")
        merged_path = MERGED_DIR / "decoder_model_merged.onnx"
    else:
        merged_path = merge_decoders_step()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean output dir
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    num_shards = quantize_to_q4(merged_path)
    print(f"\n*** Q4 merged decoder: {num_shards} shard(s) — update Constants.ts ***")

    if args.upload:
        upload_files()


if __name__ == "__main__":
    main()
