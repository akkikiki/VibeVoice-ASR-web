"""
Quantize VibeVoice-ASR ONNX models for browser deployment.

Produces quantized versions:
  - fp16: Half precision (halves size, good for WebGPU)
  - int8: 8-bit dynamic quantization (quarter size, good for WASM)
  - int4: 4-bit quantization (1/8 size, needed for 7B models in browser)

Usage:
    python quantize_onnx.py --input_dir ./onnx_output_fp32 --output_dir ./onnx_quantized
"""

import argparse
import os
import shutil
import time

import numpy as np
import onnx
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_dynamic,
)


def get_model_size(path: str) -> float:
    """Get total size of an ONNX model (graph + external data) in MB."""
    total = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / 1e6


def quantize_to_fp16(input_path: str, output_path: str):
    """Convert ONNX model from fp32 to fp16."""
    from onnxruntime.transformers import float16

    print(f"  Loading {input_path}...")

    # Load with external data
    model = onnx.load(input_path, load_external_data=True)

    print("  Converting to fp16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as fp32 for compatibility
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )

    print(f"  Saving to {output_path}...")
    # Save with external data to keep file manageable
    data_path = os.path.basename(output_path) + ".data"
    onnx.save(
        model_fp16,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
    )


def quantize_to_int8(input_path: str, output_path: str):
    """Quantize ONNX model to INT8 using dynamic quantization."""
    print(f"  Quantizing {input_path} to INT8...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )


def quantize_to_uint4(input_path: str, output_path: str):
    """Quantize ONNX model to UINT4 using dynamic quantization (MatMul only)."""
    from onnxruntime.quantization import matmul_4bits_quantizer

    print(f"  Quantizing {input_path} to 4-bit...")

    model = onnx.load(input_path, load_external_data=True)

    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model=model,
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
    )
    quant.process()

    print(f"  Saving to {output_path}...")
    data_path = os.path.basename(output_path) + ".data"
    onnx.save(
        quant.model.model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Quantize VibeVoice-ASR ONNX models")
    parser.add_argument("--input_dir", default="./onnx_output_fp32")
    parser.add_argument("--output_dir", default="./onnx_quantized")
    parser.add_argument(
        "--quantization",
        default="fp16",
        choices=["fp16", "int8", "int4", "all"],
        help="Quantization type",
    )
    args = parser.parse_args()

    models = ["speech_encoder.onnx", "decoder_with_speech.onnx", "decoder.onnx"]

    quant_types = (
        ["fp16", "int8", "int4"] if args.quantization == "all" else [args.quantization]
    )

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
                elif qtype == "int8":
                    quantize_to_int8(input_path, output_path)
                elif qtype == "int4":
                    quantize_to_uint4(input_path, output_path)

                elapsed = time.time() - t0
                new_size = get_model_size(output_path)
                ratio = new_size / orig_size * 100
                print(f"  Done in {elapsed:.1f}s — {new_size:.1f} MB ({ratio:.0f}% of original)")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

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
