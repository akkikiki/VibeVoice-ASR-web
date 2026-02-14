#!/usr/bin/env python3
"""
Export VibeVoice-ASR decoder (Qwen2 LM) with KV-cache, merge, and quantize to Q4.

Loads the Qwen2 language model part directly from VibeVoice-ASR weights,
exports prefill + decode-with-past ONNX models, merges them using optimum,
then quantizes to Q4 and reshards for browser deployment.

Usage:
    python scripts/export_and_merge_q4_kvcache.py
    python scripts/export_and_merge_q4_kvcache.py --upload
    python scripts/export_and_merge_q4_kvcache.py --skip-export  # reuse existing export
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "akkikiki/VibeVoice-ASR-onnx"
MODEL_ID = "microsoft/VibeVoice-ASR"
WORK_DIR = Path("/tmp/vibevoice-split/q4_kvcache_pipeline")
EXPORT_DIR = WORK_DIR / "onnx_export"
MERGED_DIR = WORK_DIR / "merged"
OUTPUT_DIR = WORK_DIR / "q4_output"

NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 28
NUM_KEY_VALUE_HEADS = 4
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
INTERMEDIATE_SIZE = 18944
VOCAB_SIZE = 152064

MAX_SHARD_SIZE = 1_900_000_000  # 1.9 GB
BLOCK_SIZE = 32
EMBED_TENSOR_NAME = "model.embed_tokens.weight"


# ---------------------------------------------------------------------------
# Step 1: Load Qwen2 LM from VibeVoice weights
# ---------------------------------------------------------------------------
def load_qwen2_decoder():
    """Load only the Qwen2 language model from VibeVoice-ASR."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("=" * 60)
    print("Step 1: Load Qwen2 decoder from VibeVoice-ASR")
    print("=" * 60)

    # Create Qwen2 config matching VibeVoice's decoder
    qwen2_config = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=131072,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        hidden_act="silu",
        use_cache=True,
        torch_dtype=torch.float32,
    )

    # Download safetensors
    print(f"Downloading weights from {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])
    print(f"  Downloaded to: {model_dir}")

    # Create model in bf16 to save memory (~14 GB instead of ~28 GB)
    print("Creating Qwen2ForCausalLM in bf16...")
    qwen2_config.torch_dtype = torch.bfloat16
    model = Qwen2ForCausalLM(qwen2_config).to(torch.bfloat16)

    # Load weights shard by shard, remapping "model.language_model.*" -> "model.*"
    print("Loading and remapping weights (shard by shard)...")
    import glob
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    print(f"  Found {len(safetensor_files)} safetensor files")

    loaded_count = 0
    model_state = model.state_dict()
    for sf_file in safetensor_files:
        print(f"  Loading {os.path.basename(sf_file)}...")
        shard = load_file(sf_file)
        for key, tensor in shard.items():
            new_key = None
            if key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.", 1)
            elif key.startswith("lm_head."):
                new_key = key
            if new_key and new_key in model_state:
                model_state[new_key].copy_(tensor.to(torch.bfloat16))
                loaded_count += 1
        del shard
        gc.collect()

    # Handle tied weights
    if loaded_count < len(model_state):
        if "lm_head.weight" in model_state and "model.embed_tokens.weight" in model_state:
            model_state["lm_head.weight"].copy_(model_state["model.embed_tokens.weight"])
            loaded_count += 1

    model.load_state_dict(model_state, strict=False)
    del model_state
    gc.collect()

    model.eval()

    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


# ---------------------------------------------------------------------------
# Step 2: Export to ONNX with KV-cache
# ---------------------------------------------------------------------------
class PrefillWrapper(nn.Module):
    """Prefill: input_ids + speech_embeddings -> logits + KV-cache."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # Qwen2Model
        self.lm_head = model.lm_head

    def forward(self, input_ids, speech_embeddings):
        batch_size, seq_len = input_ids.shape
        speech_len = speech_embeddings.shape[1]

        # Embed tokens
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Insert speech embeddings at positions [1 : 1+speech_len]
        inputs_embeds = torch.cat([
            inputs_embeds[:, :1, :],
            speech_embeddings,
            inputs_embeds[:, 1 + speech_len:, :],
        ], dim=1)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        # Extract KV-cache (handle both old and new DynamicCache API)
        past_kv = outputs.past_key_values
        present_tensors = []
        for i in range(NUM_HIDDEN_LAYERS):
            if hasattr(past_kv, 'key_cache'):
                present_tensors.append(past_kv.key_cache[i])
                present_tensors.append(past_kv.value_cache[i])
            elif hasattr(past_kv, 'layers'):
                present_tensors.append(past_kv.layers[i].keys)
                present_tensors.append(past_kv.layers[i].values)
            else:
                present_tensors.append(past_kv[i][0])
                present_tensors.append(past_kv[i][1])

        return (logits, *present_tensors)


class DecodeWithPastWrapper(nn.Module):
    """Decode: input_ids (1 token) + past_key_values -> logits + updated KV-cache."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, *past_key_values_flat):
        batch_size = input_ids.shape[0]

        from transformers.cache_utils import DynamicCache
        past_kv = DynamicCache()
        for i in range(NUM_HIDDEN_LAYERS):
            if hasattr(past_kv, 'update'):
                past_kv.update(past_key_values_flat[2 * i], past_key_values_flat[2 * i + 1], i)
            else:
                past_kv.key_cache.append(past_key_values_flat[2 * i])
                past_kv.value_cache.append(past_key_values_flat[2 * i + 1])

        past_len = past_key_values_flat[0].shape[2]
        position_ids = torch.tensor([[past_len]], device=input_ids.device).expand(batch_size, -1)

        inputs_embeds = self.model.embed_tokens(input_ids)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        new_past_kv = outputs.past_key_values
        present_tensors = []
        for i in range(NUM_HIDDEN_LAYERS):
            present_tensors.append(new_past_kv.layers[i].keys)
            present_tensors.append(new_past_kv.layers[i].values)

        return (logits, *present_tensors)


def export_onnx(model):
    """Export prefill + decode-with-past ONNX models."""
    print("\n" + "=" * 60)
    print("Step 2: Export ONNX models")
    print("=" * 60)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Prefill ---
    print("\nExporting prefill model...")
    prefill = PrefillWrapper(model)
    prefill.eval()

    batch_size, seq_len, speech_len = 1, 10, 5
    model_dtype = next(model.parameters()).dtype
    dummy_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    dummy_speech = torch.randn(batch_size, speech_len, HIDDEN_SIZE, dtype=model_dtype)

    input_names = ["input_ids", "speech_embeddings"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "speech_embeddings": {0: "batch_size", 1: "speech_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }
    for i in range(NUM_HIDDEN_LAYERS):
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "seq_len"}

    # Test forward
    with torch.no_grad():
        test = prefill(dummy_ids, dummy_speech)
    print(f"  Prefill test OK. logits: {test[0].shape}, KV[0]: {test[1].shape}")

    prefill_path = EXPORT_DIR / "decoder_model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            prefill, (dummy_ids, dummy_speech), str(prefill_path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=17,
            do_constant_folding=True, export_params=True,
        )
    print(f"  Saved: {prefill_path} ({prefill_path.stat().st_size / 1e6:.1f} MB)")
    del prefill
    gc.collect()

    # --- Decode with past ---
    print("\nExporting decode-with-past model...")
    decode = DecodeWithPastWrapper(model)
    decode.eval()
    model_dtype = next(model.parameters()).dtype

    dummy_ids_1 = torch.randint(0, VOCAB_SIZE, (1, 1))
    dummy_past = []
    for _ in range(NUM_HIDDEN_LAYERS):
        dummy_past.append(torch.randn(1, NUM_KEY_VALUE_HEADS, 10, HEAD_DIM, dtype=model_dtype))
        dummy_past.append(torch.randn(1, NUM_KEY_VALUE_HEADS, 10, HEAD_DIM, dtype=model_dtype))

    input_names_d = ["input_ids"]
    output_names_d = ["logits"]
    dynamic_axes_d = {"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}}
    for i in range(NUM_HIDDEN_LAYERS):
        input_names_d.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names_d.extend([f"present.{i}.key", f"present.{i}.value"])
        dynamic_axes_d[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_len"}
        dynamic_axes_d[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_len"}
        dynamic_axes_d[f"present.{i}.key"] = {0: "batch_size", 2: "total_len"}
        dynamic_axes_d[f"present.{i}.value"] = {0: "batch_size", 2: "total_len"}

    with torch.no_grad():
        test = decode(dummy_ids_1, *dummy_past)
    print(f"  Decode test OK. logits: {test[0].shape}, KV[0]: {test[1].shape}")

    decode_path = EXPORT_DIR / "decoder_with_past_model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            decode, (dummy_ids_1, *dummy_past), str(decode_path),
            input_names=input_names_d, output_names=output_names_d,
            dynamic_axes=dynamic_axes_d, opset_version=17,
            do_constant_folding=True, export_params=True,
        )
    print(f"  Saved: {decode_path} ({decode_path.stat().st_size / 1e6:.1f} MB)")
    del decode
    gc.collect()

    # List all exported files
    print("\nExported files:")
    for f in sorted(EXPORT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

    return prefill_path, decode_path


# ---------------------------------------------------------------------------
# Step 3: Merge using optimum
# ---------------------------------------------------------------------------
def merge_models(prefill_path, decode_path):
    """Merge prefill + decode into decoder_model_merged."""
    from optimum.onnx import merge_decoders

    print("\n" + "=" * 60)
    print("Step 3: Merge decoders")
    print("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prefill model...")
    prefill = onnx.load(str(prefill_path), load_external_data=True)
    print(f"  Nodes: {len(prefill.graph.node)}")

    print("Loading decode model...")
    decode = onnx.load(str(decode_path), load_external_data=True)
    print(f"  Nodes: {len(decode.graph.node)}")

    print("Merging...")
    merged_path = MERGED_DIR / "decoder_model_merged.onnx"
    merged = merge_decoders(prefill, decode, save_path=None, strict=False)
    print(f"  Merged nodes: {len(merged.graph.node)}")

    # Save
    print("Saving merged model...")
    onnx.save(
        merged, str(merged_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder_model_merged.onnx_data",
    )

    # Print I/O summary
    print(f"\nMerged model inputs ({len(merged.graph.input)}):")
    for inp in merged.graph.input[:5]:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    if len(merged.graph.input) > 5:
        print(f"  ... ({len(merged.graph.input)} total)")

    has_use_cache = any(inp.name == "use_cache_branch" for inp in merged.graph.input)
    print(f"  use_cache_branch input: {'YES' if has_use_cache else 'NO'}")

    del prefill, decode, merged
    gc.collect()

    return merged_path


# ---------------------------------------------------------------------------
# Step 4: Quantize to Q4 + reshard
# ---------------------------------------------------------------------------
def get_external_info(init):
    return {e.key: e.value for e in init.external_data}

def clear_external_data(init):
    while len(init.external_data) > 0:
        init.external_data.pop()

def set_external_data(init, location, offset, length):
    clear_external_data(init)
    init.external_data.add(key="location", value=location)
    init.external_data.add(key="offset", value=str(offset))
    init.external_data.add(key="length", value=str(length))


def quantize_q4(merged_path):
    """Quantize merged model to Q4 using MatMulNBits."""
    print("\n" + "=" * 60)
    print("Step 4: Quantize to Q4 + reshard")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean output dir
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    print("Loading merged model (protobuf only)...")
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

    init_map = {init.name: init for init in model.graph.initializer}
    matmul_weight_names = matmul_weight_names & set(init_map.keys())

    # Quantize FP32 or BF16 external weight tensors, skip embed_tokens
    fp32_weights = {}
    for name in matmul_weight_names:
        init = init_map[name]
        info = get_external_info(init)
        if "location" in info and init.data_type in (TensorProto.FLOAT, TensorProto.BFLOAT16):
            fp32_weights[name] = init

    # Don't Q4 the embedding
    for key in list(fp32_weights.keys()):
        if "embed_tokens" in key:
            del fp32_weights[key]

    print(f"  MatMul FP32 weights to quantize: {len(fp32_weights)}")

    src_data_info = {}
    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" in info:
            src_data_info[init.name] = info

    # Process each MatMul
    nodes_to_remove = []
    nodes_to_add = []
    inits_to_add = []
    inits_to_remove = set()
    quantized_count = 0

    for node in matmul_nodes:
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

        dims = list(init.dims)
        if len(dims) != 2:
            continue

        K, N = dims

        # Read weight data
        src_file = src_dir / info["location"]
        offset = int(info.get("offset", "0"))
        length = int(info["length"])
        with open(src_file, "rb") as f:
            f.seek(offset)
            raw_data = f.read(length)

        # Handle both FP32 and BF16 data
        if init.data_type == TensorProto.BFLOAT16:
            # BF16: read as uint16, convert to float32
            bf16_arr = np.frombuffer(raw_data, dtype=np.uint16)
            # BF16 to FP32: shift left by 16 bits
            fp32_bytes = (bf16_arr.astype(np.uint32) << 16).view(np.float32)
            weight_arr = fp32_bytes.reshape(K, N)
        else:
            weight_arr = np.frombuffer(raw_data, dtype=np.float32).reshape(K, N)

        # Transpose to [N, K] for column-wise block quantization
        weight_t = weight_arr.T.copy()

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

        # Quantize
        q_blocks = np.round(weight_blocks / block_scales[:, :, None]).clip(-8, 7).astype(np.int8)
        q_unsigned = (q_blocks + 8).astype(np.uint8)

        # Pack two values per byte
        packed = np.zeros((N, n_blocks_k, BLOCK_SIZE // 2), dtype=np.uint8)
        for bi in range(BLOCK_SIZE // 2):
            packed[:, :, bi] = q_unsigned[:, :, 2 * bi] | (q_unsigned[:, :, 2 * bi + 1] << 4)

        # Zero points
        zp_packed_cols = (n_blocks_k + 1) // 2
        zp_packed = np.full((N, zp_packed_cols), 0x88, dtype=np.uint8)

        b_name = f"{weight_input}_q4"
        scales_name = f"{weight_input}_scales"
        zp_name = f"{weight_input}_zp"

        inits_to_add.append(numpy_helper.from_array(packed, name=b_name))
        inits_to_add.append(numpy_helper.from_array(block_scales, name=scales_name))
        inits_to_add.append(numpy_helper.from_array(zp_packed, name=zp_name))
        inits_to_remove.add(weight_input)

        activation_input = node.input[1 - weight_idx]
        matmul_nbits = helper.make_node(
            "MatMulNBits",
            inputs=[activation_input, b_name, scales_name, zp_name],
            outputs=node.output,
            name=node.name + "_q4" if node.name else f"matmulnbits_{quantized_count}",
            domain="com.microsoft",
            K=K, N=N, bits=4, block_size=BLOCK_SIZE,
        )

        nodes_to_remove.append(node)
        nodes_to_add.append(matmul_nbits)
        quantized_count += 1

        if quantized_count % 20 == 0:
            print(f"  Quantized {quantized_count} weights...")

    print(f"  Total: {quantized_count} MatMul -> MatMulNBits")

    # Apply graph changes
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    model.graph.node.extend(nodes_to_add)

    new_inits = [init for init in model.graph.initializer if init.name not in inits_to_remove]
    new_inits.extend(inits_to_add)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Handle embed_tokens -> FP16
    for init in model.graph.initializer:
        if "embed_tokens" in init.name and init.data_type in (TensorProto.FLOAT, TensorProto.BFLOAT16):
            info = get_external_info(init)
            if "location" not in info:
                continue
            src_file = src_dir / info["location"]
            with open(src_file, "rb") as f:
                f.seek(int(info.get("offset", "0")))
                raw = f.read(int(info["length"]))
            if init.data_type == TensorProto.BFLOAT16:
                bf16_arr = np.frombuffer(raw, dtype=np.uint16)
                fp32_arr = (bf16_arr.astype(np.uint32) << 16).view(np.float32)
            else:
                fp32_arr = np.frombuffer(raw, dtype=np.float32)
            fp16_data = fp32_arr.astype(np.float16).tobytes()
            print(f"  embed_tokens: FP32 {len(raw)/1e6:.1f} MB -> FP16 {len(fp16_data)/1e6:.1f} MB")

            fp16_name = init.name + "_fp16"
            init.name = fp16_name
            init.data_type = TensorProto.FLOAT16
            clear_external_data(init)
            init.raw_data = fp16_data

            # Update Gather nodes
            for node in model.graph.node:
                if node.op_type == "Gather":
                    orig_name = fp16_name.replace("_fp16", "")
                    for idx, inp in enumerate(node.input):
                        if inp == orig_name:
                            node.input[idx] = fp16_name
                            orig_out = node.output[0]
                            node.output[0] = orig_out + "_fp16"
                            cast = helper.make_node(
                                "Cast", inputs=[orig_out + "_fp16"], outputs=[orig_out],
                                name="cast_embed_fp16", to=TensorProto.FLOAT,
                            )
                            node_idx = list(model.graph.node).index(node)
                            model.graph.node.insert(node_idx + 1, cast)
                            break
            break

    # Reshard
    print("\n  Resharding...")
    out_name = "decoder_model_merged_q4.onnx"
    shard_idx = 0
    shard_offset = 0
    shard_file = None
    shard_path = None

    def open_new_shard():
        nonlocal shard_idx, shard_offset, shard_file, shard_path
        if shard_file is not None:
            print(f"    Shard {shard_path.name}: {shard_offset / 1e6:.1f} MB")
            shard_file.close()
            shard_idx += 1
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        shard_path = OUTPUT_DIR / f"{out_name}_data{suffix}"
        shard_file = open(shard_path, "wb")
        shard_offset = 0

    def write_to_shard(data):
        nonlocal shard_offset
        if shard_offset + len(data) > MAX_SHARD_SIZE and shard_offset > 0:
            open_new_shard()
        off = shard_offset
        shard_file.write(data)
        shard_offset += len(data)
        suffix = "" if shard_idx == 0 else f"_{shard_idx}"
        return f"{out_name}_data{suffix}", off, len(data)

    open_new_shard()

    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" not in info:
            # Inline data
            if init.raw_data and len(init.raw_data) > 1024:
                loc, off, length = write_to_shard(init.raw_data)
                init.raw_data = b""
                init.data_location = TensorProto.EXTERNAL
                set_external_data(init, loc, off, length)
            continue

        # External data
        src_file = src_dir / info["location"]
        src_offset = int(info.get("offset", "0"))
        src_length = int(info["length"])
        with open(src_file, "rb") as f:
            f.seek(src_offset)
            data = f.read(src_length)
        loc, off, length = write_to_shard(data)
        set_external_data(init, loc, off, length)

    if shard_file is not None:
        print(f"    Shard {shard_path.name}: {shard_offset / 1e6:.1f} MB")
        shard_file.close()
    num_shards = shard_idx + 1

    # Save protobuf
    out_path = OUTPUT_DIR / out_name
    onnx.save(model, str(out_path))

    print(f"\nOutput files ({num_shards} shards):")
    total = 0
    for f in sorted(OUTPUT_DIR.iterdir()):
        sz = f.stat().st_size / 1e6
        total += sz
        print(f"  {f.name}: {sz:.1f} MB")
    print(f"  Total: {total / 1024:.2f} GB")

    return num_shards


# ---------------------------------------------------------------------------
# Step 5: Upload
# ---------------------------------------------------------------------------
def upload():
    from huggingface_hub import HfApi
    api = HfApi()

    print(f"\n{'='*60}")
    print("Step 5: Upload to HuggingFace")
    print(f"{'='*60}")

    for f in sorted(OUTPUT_DIR.iterdir()):
        remote = f"onnx/{f.name}"
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB) -> {remote}")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=remote, repo_id=REPO_ID)
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing ONNX export")
    args = parser.parse_args()

    if args.skip_export and (EXPORT_DIR / "decoder_model.onnx").exists():
        print("Reusing existing ONNX export.")
        prefill_path = EXPORT_DIR / "decoder_model.onnx"
        decode_path = EXPORT_DIR / "decoder_with_past_model.onnx"
    else:
        model = load_qwen2_decoder()
        prefill_path, decode_path = export_onnx(model)
        del model
        gc.collect()

    merged_path = merge_models(prefill_path, decode_path)
    num_shards = quantize_q4(merged_path)
    print(f"\n*** Q4 KV-cache merged decoder: {num_shards} shards ***")
    print("*** Update Constants.ts with new shard count ***")

    if args.upload:
        upload()


if __name__ == "__main__":
    main()
