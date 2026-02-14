#!/usr/bin/env python3
"""
Convert FP32 speech encoder weights to FP16 with Cast nodes.

Strategy: For each FP32 initializer, convert to FP16 and insert a
Cast(FP16->FP32) node after the weight is loaded. This way:
- Storage is FP16 (smaller download)
- All ops receive FP32 inputs (no type mismatch)

This is the same approach used for the decoder's embed_tokens.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
WORK_DIR = Path("/tmp/vibevoice-split/encoder_fp16")
OUTPUT_NAME = "encoder_model_fp16"


def download_encoder():
    from huggingface_hub import hf_hub_download
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading speech_encoder.onnx...")
    hf_hub_download(REPO_ID, "onnx/speech_encoder.onnx", local_dir=WORK_DIR)
    print("Downloading speech_encoder.onnx.data...")
    hf_hub_download(REPO_ID, "onnx/speech_encoder.onnx.data", local_dir=WORK_DIR)
    return WORK_DIR / "onnx" / "speech_encoder.onnx"


def convert_to_fp16_with_cast(model_path: Path):
    """Convert FP32 weights to FP16 + Cast nodes."""
    out_dir = WORK_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean output
    for f in out_dir.iterdir():
        f.unlink()

    print(f"Loading {model_path.name} (protobuf only)...")
    model = onnx.load(str(model_path), load_external_data=False)
    src_dir = model_path.parent

    # Build map: initializer name -> list of (node, input_index) that use it
    init_names = {init.name for init in model.graph.initializer}

    # Find FP32 initializers with external data
    fp32_inits = []
    for init in model.graph.initializer:
        info = {e.key: e.value for e in init.external_data}
        if "location" not in info:
            continue
        if init.data_type == TensorProto.FLOAT:
            fp32_inits.append({
                "init": init,
                "location": info["location"],
                "offset": int(info.get("offset", "0")),
                "length": int(info["length"]),
            })

    print(f"FP32 initializers to convert: {len(fp32_inits)}")

    # For each FP32 initializer:
    # 1. Rename to {name}_fp16
    # 2. Change data_type to FLOAT16
    # 3. Insert Cast node: {name}_fp16 -> {name} (Cast to FLOAT)
    cast_nodes = []
    converted_names = set()

    for t in fp32_inits:
        init = t["init"]
        original_name = init.name
        fp16_name = original_name + "_fp16"

        # Rename initializer
        init.name = fp16_name
        init.data_type = TensorProto.FLOAT16
        converted_names.add(original_name)

        # Create Cast node
        cast_node = helper.make_node(
            "Cast",
            inputs=[fp16_name],
            outputs=[original_name],
            name=f"cast_{original_name.replace('.', '_')}_fp16_to_fp32",
            to=TensorProto.FLOAT,
        )
        cast_nodes.append(cast_node)

    # Insert all Cast nodes at the beginning of the graph
    for i, cast_node in enumerate(cast_nodes):
        model.graph.node.insert(i, cast_node)

    print(f"Inserted {len(cast_nodes)} Cast nodes")

    # Now write the external data file with FP16 converted weights
    data_filename = f"{OUTPUT_NAME}.onnx_data"
    data_path = out_dir / data_filename
    offset = 0

    total_fp32_size = 0
    total_fp16_size = 0

    with open(data_path, "wb") as out_f:
        for init in model.graph.initializer:
            info = {e.key: e.value for e in init.external_data}
            if "location" not in info:
                continue

            src_file = src_dir / info["location"]
            src_offset = int(info.get("offset", "0"))
            src_length = int(info["length"])

            # Read source data
            with open(src_file, "rb") as f:
                f.seek(src_offset)
                raw = f.read(src_length)

            # Convert if this was an FP32 init (now marked as FP16)
            original_name = init.name.removesuffix("_fp16")
            if original_name in converted_names and init.name.endswith("_fp16"):
                # Convert FP32 bytes -> FP16 bytes
                arr = np.frombuffer(raw, dtype=np.float32)
                data = arr.astype(np.float16).tobytes()
                total_fp32_size += len(raw)
                total_fp16_size += len(data)
            else:
                data = raw

            # Update external data references
            while len(init.external_data) > 0:
                init.external_data.pop()
            init.external_data.add(key="location", value=data_filename)
            init.external_data.add(key="offset", value=str(offset))
            init.external_data.add(key="length", value=str(len(data)))

            out_f.write(data)
            offset += len(data)

    print(f"FP32 -> FP16: {total_fp32_size / (1024**2):.1f} MB -> {total_fp16_size / (1024**2):.1f} MB")
    print(f"Total data file: {offset / (1024**2):.1f} MB")

    # Save ONNX protobuf
    out_onnx = out_dir / f"{OUTPUT_NAME}.onnx"
    onnx.save(model, str(out_onnx))

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.skip_download:
        model_path = WORK_DIR / "onnx" / "speech_encoder.onnx"
    else:
        model_path = download_encoder()

    out_dir = convert_to_fp16_with_cast(model_path)

    print(f"\nOutput files:")
    total = 0
    for f in sorted(out_dir.iterdir()):
        size_mb = f.stat().st_size / (1024**2)
        total += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  Total: {total:.1f} MB")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for f in sorted(out_dir.iterdir()):
            remote_path = f"onnx/{f.name}"
            print(f"Uploading {f.name} -> {remote_path}...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
            )
        print("Upload complete!")
    else:
        print("\nTo upload, run again with --upload flag.")


if __name__ == "__main__":
    main()
