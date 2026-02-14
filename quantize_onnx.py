"""
Quantize VibeVoice-ASR ONNX models for browser deployment.

Produces quantized versions:
  - fp16: Half precision (halves size, good for WebGPU)
  - int4: 4-bit block-wise quantization (MatMulNBits, ~6x compression)

The int4 quantization uses a two-pass approach to handle large models:
  1. Cast large fp32 weights to fp16 (avoids protobuf 2GB limit)
  2. Quantize MatMul weights to int4 with block-wise symmetric quantization

Usage:
    python quantize_onnx.py --input_dir ./onnx_output_fp32 --output_dir ./onnx_quantized
    python quantize_onnx.py --quantization int4
    python quantize_onnx.py --quantization fp16
"""

import argparse
import gc
import os
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data, uses_external_data


def get_model_size(path: str) -> float:
    """Get total size of an ONNX model (graph + external data) in MB."""
    total = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / 1e6


def save_with_external_data(model, output_path: str, size_threshold: int = 1024):
    """Save ONNX model with proper external data handling.

    Uses convert_model_to_external_data before save to avoid the bloated
    output that onnx.save(..., save_as_external_data=True) can produce.
    """
    # Strip any existing external data references (data already in raw_data)
    for tensor in model.graph.initializer:
        if uses_external_data(tensor):
            while len(tensor.external_data) > 0:
                tensor.external_data.pop()
            tensor.data_location = 0

    data_location = os.path.basename(output_path) + ".data"

    # Remove old data file if it exists
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_location,
        size_threshold=size_threshold,
    )
    onnx.save(model, output_path)


# ---------------------------------------------------------------------------
# fp16 quantization
# ---------------------------------------------------------------------------

def quantize_to_fp16(input_path: str, output_path: str):
    """Convert ONNX model from fp32 to fp16 by casting weight tensors.

    Note: The onnxruntime.transformers.float16 converter can produce empty
    graphs for models with Conv1d + GroupNorm (e.g., VAE encoders). This
    function uses manual numpy casting instead, which preserves all nodes.
    """
    print(f"  Loading {input_path}...")
    model = onnx.load(input_path, load_external_data=True)

    cast_count = 0
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(list(tensor.dims))
            new_tensor = numpy_helper.from_array(arr.astype(np.float16), tensor.name)
            tensor.CopyFrom(new_tensor)
            cast_count += 1
            del arr

    print(f"  Cast {cast_count} fp32 tensors to fp16")
    print(f"  Saving to {output_path}...")
    save_with_external_data(model, output_path)


# ---------------------------------------------------------------------------
# int4 quantization
# ---------------------------------------------------------------------------

def quantize_weight_int4(weight: np.ndarray, block_size: int = 32):
    """Block-wise symmetric int4 quantization for MatMulNBits.

    MatMulNBits computes A @ B where B has shape (K, N).
    Quantization is along the K dimension (contraction dim).
    Packed data is stored transposed as (N, ...) per the onnxruntime spec:
      - packed: (N, ceil(K/block_size) * block_size / 2) as uint8
      - scales: (N, ceil(K/block_size)) as float32
      - zero_points: (N, ceil(ceil(K/block_size)/2)) as packed uint8

    Args:
        weight: fp32 weight tensor of shape (K, N)
        block_size: number of elements per quantization block

    Returns:
        packed: uint8 array with two int4 values packed per byte
        scales: fp32 per-block scale factors
        zero_points_packed: uint8 packed zero points (always 8 for symmetric)
        K, N: original dimensions
    """
    K, N = weight.shape

    # Transpose to (N, K) — MatMulNBits stores data in N-major order
    weight_t = weight.T  # (N, K)

    # Pad K to multiple of block_size
    pad_k = (block_size - K % block_size) % block_size
    if pad_k > 0:
        weight_t = np.pad(weight_t, ((0, 0), (0, pad_k)))

    n_blocks = weight_t.shape[1] // block_size
    weight_blocked = weight_t.reshape(N, n_blocks, block_size)

    # Symmetric quantization: range [-8, 7] maps to [-scale*8, scale*7]
    abs_max = np.abs(weight_blocked).max(axis=2)  # (N, n_blocks)
    scales = np.where(abs_max == 0, 1e-6, abs_max / 7.0).astype(np.float32)

    weight_scaled = weight_blocked / scales[:, :, np.newaxis]
    weight_int = np.clip(np.round(weight_scaled), -8, 7).astype(np.int8)

    # Shift to unsigned [0, 15] for uint4
    weight_uint4 = (weight_int + 8).astype(np.uint8)
    weight_flat = weight_uint4.reshape(N, -1)  # (N, K_padded)

    # Pack two uint4 values into one uint8 (low nibble first)
    packed = weight_flat[:, 0::2] | (weight_flat[:, 1::2] << 4)

    # Zero points (always 8 for symmetric quantization)
    zp = np.full((N, n_blocks), 8, dtype=np.uint8)
    if n_blocks % 2 == 1:
        zp = np.pad(zp, ((0, 0), (0, 1)))
    zp_packed = zp[:, 0::2] | (zp[:, 1::2] << 4)

    return packed, scales, zp_packed, K, N


def quantize_to_int4(input_path: str, output_path: str, block_size: int = 32,
                     min_weight_size_mb: float = 1.0):
    """Quantize ONNX model MatMul weights to int4 using MatMulNBits op.

    Two-pass approach for large models (>2GB weights):
      Pass 1: Cast all large fp32 weights to fp16 (avoids protobuf 2GB limit)
      Pass 2: Quantize MatMul weights to int4 with block-wise symmetric quant

    Non-MatMul weights (embeddings, LayerNorm, etc.) are kept in fp16.
    """
    min_bytes = int(min_weight_size_mb * 1e6)

    # Check if any tensor exceeds 2GB (needs two-pass approach)
    model = onnx.load(input_path, load_external_data=True)
    max_tensor_size = max(len(t.raw_data) for t in model.graph.initializer)
    needs_two_pass = max_tensor_size > 2_000_000_000

    if needs_two_pass:
        print("  Large tensors detected (>2GB), using two-pass approach...")
        # Pass 1: Cast large fp32 to fp16, save to temp file
        for tensor in model.graph.initializer:
            if tensor.data_type == TensorProto.FLOAT and len(tensor.raw_data) > 100_000_000:
                arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(list(tensor.dims))
                new_t = numpy_helper.from_array(arr.astype(np.float16), tensor.name)
                tensor.CopyFrom(new_t)
                del arr

        tmp_path = output_path + ".tmp"
        onnx.save(model, tmp_path, save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(tmp_path) + ".data",
                  size_threshold=1024)
        del model
        gc.collect()

        # Reload from temp
        model = onnx.load(tmp_path, load_external_data=True)

        # Clean up temp files
        for f in [tmp_path, tmp_path + ".data"]:
            if os.path.exists(f):
                os.remove(f)

    # Cast any remaining fp32 tensors to fp16
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(list(tensor.dims))
            new_t = numpy_helper.from_array(arr.astype(np.float16), tensor.name)
            tensor.CopyFrom(new_t)
            del arr

    # Find MatMul nodes with weight as second input
    init_map = {t.name: t for t in model.graph.initializer}
    matmul_weights = set()
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.input[1] in init_map:
            if len(init_map[node.input[1]].raw_data) > min_bytes:
                matmul_weights.add(node.input[1])

    print(f"  Found {len(matmul_weights)} MatMul weights to quantize")

    total_orig = 0
    total_quantized = 0
    nodes_to_replace = []
    new_initializers = []
    removed_names = set()

    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in matmul_weights:
            continue

        weight_init = init_map[node.input[1]]
        weight_arr = numpy_helper.to_array(weight_init).astype(np.float32)
        if weight_arr.ndim != 2:
            continue

        K, N = weight_arr.shape
        total_orig += weight_arr.nbytes

        packed, scales, zp_packed, _, _ = quantize_weight_int4(weight_arr, block_size)
        total_quantized += packed.nbytes + scales.nbytes + zp_packed.nbytes

        prefix = weight_init.name.replace(".", "_")
        new_initializers.append(numpy_helper.from_array(packed, f"{prefix}_Q4"))
        new_initializers.append(numpy_helper.from_array(scales, f"{prefix}_scales"))
        new_initializers.append(numpy_helper.from_array(zp_packed, f"{prefix}_zp"))
        removed_names.add(weight_init.name)

        new_node = helper.make_node(
            "MatMulNBits",
            inputs=[node.input[0], f"{prefix}_Q4", f"{prefix}_scales", f"{prefix}_zp"],
            outputs=node.output,
            name=node.name + "_int4" if node.name else "",
            domain="com.microsoft",
            K=K, N=N, bits=4, block_size=block_size,
        )
        nodes_to_replace.append((node, new_node))
        del weight_arr, packed, scales, zp_packed

    # Apply node replacements
    for old_node, new_node in nodes_to_replace:
        model.graph.node.remove(old_node)
        model.graph.node.append(new_node)

    # Update initializers
    keep = [i for i in model.graph.initializer if i.name not in removed_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    model.graph.initializer.extend(new_initializers)

    # Add com.microsoft opset for MatMulNBits
    if not any(op.domain == "com.microsoft" for op in model.opset_import):
        ms_opset = onnx.OperatorSetIdProto()
        ms_opset.domain = "com.microsoft"
        ms_opset.version = 1
        model.opset_import.append(ms_opset)

    compression = total_orig / max(total_quantized, 1)
    print(f"  MatMul weights: {total_orig/1e9:.2f} GB -> {total_quantized/1e9:.2f} GB ({compression:.1f}x)")

    print(f"  Saving to {output_path}...")
    save_with_external_data(model, output_path)

    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantize VibeVoice-ASR ONNX models")
    parser.add_argument("--input_dir", default="./onnx_output_fp32")
    parser.add_argument("--output_dir", default="./onnx_quantized")
    parser.add_argument(
        "--quantization",
        default="int4",
        choices=["fp16", "int4", "all"],
        help="Quantization type (default: int4)",
    )
    parser.add_argument("--block_size", type=int, default=32, help="Int4 block size")
    args = parser.parse_args()

    models = ["speech_encoder.onnx", "decoder_with_speech.onnx", "decoder.onnx"]
    quant_types = ["fp16", "int4"] if args.quantization == "all" else [args.quantization]

    for qtype in quant_types:
        out_dir = os.path.join(args.output_dir, qtype)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Quantizing to {qtype}")
        print(f"{'='*60}")

        for model_name in models:
            input_path = os.path.join(args.input_dir, model_name)
            if not os.path.exists(input_path):
                print(f"  Skipping {model_name} (not found)")
                continue

            output_path = os.path.join(out_dir, model_name)
            orig_size = get_model_size(input_path)
            print(f"\n  {model_name} (original: {orig_size:.1f} MB)")

            t0 = time.time()
            try:
                if qtype == "fp16":
                    quantize_to_fp16(input_path, output_path)
                elif qtype == "int4":
                    quantize_to_int4(input_path, output_path, block_size=args.block_size)

                elapsed = time.time() - t0
                new_size = get_model_size(output_path)
                ratio = new_size / orig_size * 100
                print(f"  Done in {elapsed:.1f}s — {new_size:.1f} MB ({ratio:.0f}% of original)")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for qtype in quant_types:
        out_dir = os.path.join(args.output_dir, qtype)
        if not os.path.exists(out_dir):
            continue
        print(f"\n  {qtype}:")
        total = 0
        for f in sorted(os.listdir(out_dir)):
            fp = os.path.join(out_dir, f)
            size = os.path.getsize(fp) / 1e6
            total += size
            print(f"    {f}: {size:.1f} MB")
        print(f"    TOTAL: {total:.1f} MB")


if __name__ == "__main__":
    main()
