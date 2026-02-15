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
    # Use eager attention (not SDPA) so torch.onnx.export works without
    # hitting the "GuardOnDataDependentSymNode" error in SDPA's conditional.
    print("Creating Qwen2ForCausalLM in bf16 (eager attention)...")
    qwen2_config.torch_dtype = torch.bfloat16
    qwen2_config._attn_implementation = "eager"
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
                model_state[new_key].copy_(tensor.to(torch.float32))
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

        # Position IDs must match actual sequence length after speech insertion
        actual_seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(actual_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

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
            if hasattr(new_past_kv, 'key_cache'):
                present_tensors.append(new_past_kv.key_cache[i])
                present_tensors.append(new_past_kv.value_cache[i])
            elif hasattr(new_past_kv, 'layers'):
                present_tensors.append(new_past_kv.layers[i].keys)
                present_tensors.append(new_past_kv.layers[i].values)
            else:
                present_tensors.append(new_past_kv[i][0])
                present_tensors.append(new_past_kv[i][1])

        return (logits, *present_tensors)


def _fix_causal_mask_constants(model_proto, traced_seq_len, source_input=None,
                                source_dim=None, add_offset=0):
    """Replace hardcoded seq_len constants in causal mask with dynamic computations.

    During torch.onnx.export, Qwen2's _update_causal_mask bakes the traced
    seq_len into Constant nodes for Reshape/Equal/Where ops. This function
    finds those constants and replaces them with dynamic computations.

    Strategy:
    1. Try to reuse the existing dynamic seq_len already in the graph by
       tracing backwards from Reshape nodes that consume the hardcoded constants
       (prefill model — seq_len comes from intermediate Concat output).
    2. If no Reshape consumer found, fall back to creating Shape(source_input)
       → Gather(source_dim) nodes (decode model — past_len/total_len from
       past_key_values shape).
    """
    import numpy as np

    # Build lookup maps
    output_to_node = {}
    input_to_consumers = {}
    for node in model_proto.graph.node:
        for out in node.output:
            output_to_node[out] = node
        for inp in node.input:
            if inp not in input_to_consumers:
                input_to_consumers[inp] = []
            input_to_consumers[inp].append(node)

    # Find Constant nodes with hardcoded traced_seq_len
    constants_to_fix = []
    for node in model_proto.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.t and attr.t.data_type == 7:  # int64
                    if attr.t.raw_data:
                        arr = np.frombuffer(attr.t.raw_data, dtype=np.int64)
                    else:
                        arr = np.array(attr.t.int64_data, dtype=np.int64)
                    if len(arr) > 0 and traced_seq_len in arr:
                        constants_to_fix.append((node, list(arr)))

    if not constants_to_fix:
        print(f"  No hardcoded constants with value {traced_seq_len} found")
        return

    print(f"  Fixing {len(constants_to_fix)} hardcoded seq_len={traced_seq_len} constants")

    # Find the existing dynamic seq_len by tracing from the Reshape that uses
    # one of our hardcoded constants:
    #   Reshape(data=Range_output, shape=hardcoded_constant)
    #   → Range has limit = Cast(Gather(Shape(intermediate)))
    #   → We want the Range's limit (before Cast), which is the dynamic seq_len
    dynamic_scalar = None
    tag = f"_fix_{traced_seq_len}"

    for const_node, arr in constants_to_fix:
        const_out = const_node.output[0]
        for consumer in input_to_consumers.get(const_out, []):
            if consumer.op_type == "Reshape":
                # Found Reshape consuming our constant as shape.
                # The data input should come from a Range node.
                data_input = consumer.input[0]
                range_node = output_to_node.get(data_input)
                if range_node and range_node.op_type == "Range":
                    # Range(start, limit, step) — limit is the dynamic seq_len
                    limit_name = range_node.input[1]
                    # limit typically comes from Cast(Gather(...))
                    cast_node = output_to_node.get(limit_name)
                    if cast_node and cast_node.op_type == "Cast":
                        # The Cast input is the int64 dynamic scalar
                        dynamic_scalar = cast_node.input[0]
                    else:
                        # Maybe limit is directly the dynamic scalar
                        dynamic_scalar = limit_name
                    break
        if dynamic_scalar:
            break

    if dynamic_scalar is None and source_input is not None and source_dim is not None:
        # Fallback: create explicit Shape → Gather to extract dim from source_input
        print(f"  No Reshape consumer found, using explicit Shape({source_input})[{source_dim}]")
        shape_out = f"{tag}_shape"
        gather_out = f"{tag}_dim"
        dim_idx_name = f"{tag}_dim_idx"

        shape_node = helper.make_node(
            "Shape", inputs=[source_input], outputs=[shape_out],
            name=f"{tag}/Shape",
        )
        dim_idx_init = numpy_helper.from_array(
            np.array(source_dim, dtype=np.int64), name=dim_idx_name
        )
        gather_node = helper.make_node(
            "Gather", inputs=[shape_out, dim_idx_name], outputs=[gather_out],
            name=f"{tag}/Gather", axis=0,
        )
        model_proto.graph.initializer.append(dim_idx_init)
        nodes_to_add_prefix = [shape_node, gather_node]

        dynamic_scalar = gather_out
        if add_offset != 0:
            offset_name = f"{tag}_offset"
            add_out = f"{tag}_added"
            offset_init = numpy_helper.from_array(
                np.array(add_offset, dtype=np.int64), name=offset_name
            )
            add_node = helper.make_node(
                "Add", inputs=[gather_out, offset_name], outputs=[add_out],
                name=f"{tag}/Add",
            )
            model_proto.graph.initializer.append(offset_init)
            nodes_to_add_prefix.append(add_node)
            dynamic_scalar = add_out
    else:
        nodes_to_add_prefix = []

    if dynamic_scalar is None:
        print("  WARNING: Could not find dynamic seq_len, skipping fixup")
        return

    print(f"  Using dynamic scalar: {dynamic_scalar}")

    # Reshape the scalar to [1] tensor for use with Concat
    seq_len_1d_name = f"{tag}_1d"
    shape_1_name = f"{tag}_shape_1"
    nodes_to_add = list(nodes_to_add_prefix)

    reshape_node = helper.make_node(
        "Reshape", inputs=[dynamic_scalar, shape_1_name], outputs=[seq_len_1d_name],
        name=f"{tag}/Reshape_1d",
    )
    shape_1_init = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=shape_1_name
    )
    model_proto.graph.initializer.append(shape_1_init)
    nodes_to_add.append(reshape_node)

    for i, (const_node, arr) in enumerate(constants_to_fix):
        out_name = const_node.output[0]

        # Build dynamic shape by replacing traced_seq_len with seq_len_1d
        # e.g. [10, 1] → Concat([seq_len_1d, [1]])
        # e.g. [1, 10, 10] → Concat([[1], seq_len_1d, seq_len_1d])
        concat_inputs = []
        static_parts = []

        for val in arr:
            if val == traced_seq_len:
                if static_parts:
                    static_name = f"{tag}_static_{i}_{len(concat_inputs)}"
                    static_init = numpy_helper.from_array(
                        np.array(static_parts, dtype=np.int64), name=static_name
                    )
                    model_proto.graph.initializer.append(static_init)
                    concat_inputs.append(static_name)
                    static_parts = []
                concat_inputs.append(seq_len_1d_name)
            else:
                static_parts.append(val)

        if static_parts:
            static_name = f"{tag}_static_{i}_{len(concat_inputs)}"
            static_init = numpy_helper.from_array(
                np.array(static_parts, dtype=np.int64), name=static_name
            )
            model_proto.graph.initializer.append(static_init)
            concat_inputs.append(static_name)

        # Replace the Constant node with a Concat node producing the same output
        concat_node = helper.make_node(
            "Concat", inputs=concat_inputs, outputs=[out_name],
            name=f"{tag}/Concat_{i}", axis=0,
        )
        nodes_to_add.append(concat_node)

        # Remove the old Constant node
        model_proto.graph.node.remove(const_node)
        print(f"    {const_node.name}: {arr} → dynamic")

    # Insert fixup nodes after the producer of dynamic_scalar (topological order)
    insert_idx = 0
    if dynamic_scalar:
        for i, node in enumerate(model_proto.graph.node):
            if dynamic_scalar in node.output:
                insert_idx = i + 1
                break
    for j, node in enumerate(nodes_to_add):
        model_proto.graph.node.insert(insert_idx + j, node)


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

    batch_size, seq_len, speech_len = 1, 73, 37
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
            dynamo=False,  # Use legacy tracer-based exporter
        )
    # Fix hardcoded causal mask constants before consolidating
    print("  Fixing hardcoded causal mask constants...")
    prefill_proto = onnx.load(str(prefill_path), load_external_data=False)
    _fix_causal_mask_constants(prefill_proto, traced_seq_len=seq_len)
    onnx.save(prefill_proto, str(prefill_path))
    del prefill_proto

    # Re-save with consolidated external data for merge compatibility
    print("  Consolidating external data...")
    prefill_model = onnx.load(str(prefill_path), load_external_data=True)
    onnx.save(
        prefill_model, str(prefill_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder_model.onnx_data",
    )
    del prefill_model
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
        dummy_past.append(torch.randn(1, NUM_KEY_VALUE_HEADS, 73, HEAD_DIM, dtype=model_dtype))
        dummy_past.append(torch.randn(1, NUM_KEY_VALUE_HEADS, 73, HEAD_DIM, dtype=model_dtype))

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
            dynamo=False,  # Use legacy tracer-based exporter
        )
    # Fix hardcoded causal mask constants (past_len=73 → total_len=74)
    print("  Fixing hardcoded causal mask constants...")
    decode_proto = onnx.load(str(decode_path), load_external_data=False)
    _fix_causal_mask_constants(decode_proto, traced_seq_len=73,
                               source_input="past_key_values.0.key", source_dim=2)  # past_len
    _fix_causal_mask_constants(decode_proto, traced_seq_len=74,
                               source_input="past_key_values.0.key", source_dim=2,
                               add_offset=1)  # total_len = past_len + 1
    onnx.save(decode_proto, str(decode_path))
    del decode_proto

    # Re-save with consolidated external data for merge compatibility
    print("  Consolidating external data...")
    decode_model = onnx.load(str(decode_path), load_external_data=True)
    onnx.save(
        decode_model, str(decode_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder_with_past_model.onnx_data",
    )
    del decode_model
    print(f"  Saved: {decode_path} ({decode_path.stat().st_size / 1e6:.1f} MB)")
    del decode
    gc.collect()

    # List all exported files
    print("\nExported files:")
    for f in sorted(EXPORT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

    return prefill_path, decode_path


# ---------------------------------------------------------------------------
# Step 3: Merge prefill + decode with If node (protobuf-only, no data load)
# ---------------------------------------------------------------------------
def merge_models(prefill_path, decode_path):
    """
    Merge prefill + decode into a single model with an If node, replicating
    what optimum.onnx.merge_decoders does but without loading ~30GB external
    data into memory.

    Both models share the same Qwen2 weights but torch.onnx.export assigns
    different onnx::MatMul_* names. We deduplicate by matching node names
    between graphs, so the merged model uses only prefill's external data file.
    """
    import copy

    print("\n" + "=" * 60)
    print("Step 3: Merge prefill + decode models (If node)")
    print("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Load both models without external data (~1.3 MB each)
    print("Loading model protos (no external data)...")
    prefill = onnx.load(str(prefill_path), load_external_data=False)
    decode = onnx.load(str(decode_path), load_external_data=False)

    print(f"  Prefill: {len(prefill.graph.node)} nodes, {len(prefill.graph.initializer)} inits")
    print(f"  Decode:  {len(decode.graph.node)} nodes, {len(decode.graph.initializer)} inits")

    # --- Classify initializers ---
    p_inits = {i.name: i for i in prefill.graph.initializer}
    d_inits = {i.name: i for i in decode.graph.initializer}

    def is_small_init(init):
        """Scalar or 1-D int initializers stay in subgraphs (optimum convention)."""
        return len(init.dims) == 0 or (len(init.dims) == 1 and init.data_type in [6, 7])

    # Truly shared: same name, same dims, same type
    truly_shared = set()
    name_collisions = set()
    for name in set(p_inits) & set(d_inits):
        if (tuple(p_inits[name].dims) == tuple(d_inits[name].dims) and
                p_inits[name].data_type == d_inits[name].data_type):
            truly_shared.add(name)
        else:
            name_collisions.add(name)

    print(f"  Truly shared: {len(truly_shared)}, Name collisions: {len(name_collisions)}")

    # --- Build decode → prefill initializer remapping ---
    # Match large initializers by (node_name, input_idx) between graphs
    def build_large_init_usage(graph, init_map):
        usage = {}
        for node in graph.node:
            for idx, inp in enumerate(node.input):
                if inp in init_map and not is_small_init(init_map[inp]):
                    usage[(node.name, idx)] = inp
        return usage

    p_usage = build_large_init_usage(prefill.graph, p_inits)
    d_usage = build_large_init_usage(decode.graph, d_inits)

    # Identity mapping for truly shared large initializers
    remap = {}
    for name in truly_shared:
        if name in d_inits and not is_small_init(d_inits[name]):
            remap[name] = name

    # Match non-shared by node name + input position
    for (node_name, inp_idx), d_init_name in d_usage.items():
        if d_init_name not in remap:
            p_init_name = p_usage.get((node_name, inp_idx))
            if p_init_name:
                remap[d_init_name] = p_init_name

    remapped_count = sum(1 for k, v in remap.items() if k != v)
    print(f"  Remapped {remapped_count} decode initializers to prefill equivalents")

    unmapped = [n for n in d_inits if not is_small_init(d_inits[n]) and n not in remap]
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped large decode initializers: {unmapped[:3]}")

    # --- Apply remapping to decode graph nodes ---
    for node in decode.graph.node:
        for idx in range(len(node.input)):
            if node.input[idx] in remap:
                node.input[idx] = remap[node.input[idx]]

    # --- Unify outputs ---
    # Both models have the same output names; use prefill's shape specs
    p_out_map = {o.name: o for o in prefill.graph.output}
    unified_d_outputs = []
    for d_out in decode.graph.output:
        if d_out.name in p_out_map:
            unified_d_outputs.append(copy.deepcopy(p_out_map[d_out.name]))
        else:
            unified_d_outputs.append(d_out)

    # --- Separate small vs large initializers ---
    prefill_small = [i for i in prefill.graph.initializer if is_small_init(i)]
    decode_small = [i for i in decode.graph.initializer if is_small_init(i)]
    large_inits = [i for i in prefill.graph.initializer if not is_small_init(i)]

    # Update external data location to merged filename
    merged_data_name = "decoder_model_merged.onnx_data"
    for init in large_inits:
        for ext in init.external_data:
            if ext.key == "location":
                ext.value = merged_data_name

    print(f"  Large (top-level): {len(large_inits)}, "
          f"Small (prefill sub): {len(prefill_small)}, Small (decode sub): {len(decode_small)}")

    # --- Build subgraphs ---
    no_past_branch = helper.make_graph(
        nodes=prefill.graph.node,
        name="no_past",
        inputs=[],
        outputs=list(prefill.graph.output),
        initializer=prefill_small,
    )

    with_past_branch = helper.make_graph(
        nodes=decode.graph.node,
        name="with_past",
        inputs=[],
        outputs=unified_d_outputs,
        initializer=decode_small,
    )

    # --- Union of all inputs + use_cache_branch ---
    all_inputs = []
    seen = set()
    for inp in list(prefill.graph.input) + list(decode.graph.input):
        if inp.name not in seen:
            all_inputs.append(inp)
            seen.add(inp.name)

    use_cache_branch = helper.make_tensor_value_info(
        name="use_cache_branch", elem_type=TensorProto.BOOL, shape=[1],
    )

    # --- If node: true → with_past, false → no_past ---
    if_node = helper.make_node(
        "If",
        inputs=["use_cache_branch"],
        outputs=[o.name for o in no_past_branch.output],
        name="optimum::if",
        then_branch=with_past_branch,
        else_branch=no_past_branch,
    )

    merged_graph = helper.make_graph(
        nodes=[if_node],
        name="merged",
        inputs=[*all_inputs, use_cache_branch],
        outputs=list(no_past_branch.output),
        initializer=large_inits,
    )

    # Preserve opset imports from both models
    opset_imports = []
    opset_domains = set()
    for oi in list(prefill.opset_import) + list(decode.opset_import):
        if oi.domain not in opset_domains:
            opset_imports.append(oi)
            opset_domains.add(oi.domain)

    merged_model = helper.make_model_gen_version(
        merged_graph, producer_name="optimum-onnx",
        opset_imports=opset_imports, ir_version=9,
    )

    # Save merged protobuf
    merged_path = MERGED_DIR / "decoder_model_merged.onnx"
    onnx.save(merged_model, str(merged_path))

    # Symlink prefill's external data
    prefill_data = prefill_path.parent / "decoder_model.onnx_data"
    merged_data = MERGED_DIR / merged_data_name
    if merged_data.exists():
        merged_data.unlink()
    os.symlink(str(prefill_data.resolve()), str(merged_data))

    # --- Verify ---
    m = onnx.load(str(merged_path), load_external_data=False)
    input_names = [i.name for i in m.graph.input]
    output_names = [o.name for o in m.graph.output]
    proto_mb = merged_path.stat().st_size / 1e6

    print(f"\n  Merged proto: {proto_mb:.1f} MB")
    print(f"  Inputs ({len(input_names)}): {input_names[:3]} ... {input_names[-1]}")
    print(f"  Outputs ({len(output_names)}): {output_names[:2]} ... {output_names[-1]}")
    print(f"  use_cache_branch: {'use_cache_branch' in input_names}")
    print(f"  past_key_values:  {any(n.startswith('past_key_values') for n in input_names)}")
    print(f"  speech_embeddings: {'speech_embeddings' in input_names}")

    del prefill, decode, merged_model, m
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


def _get_subgraphs(model):
    """Extract If-node subgraphs, or return [model.graph] for flat models."""
    for node in model.graph.node:
        if node.op_type == "If":
            sgs = {}
            for attr in node.attribute:
                if attr.name in ("then_branch", "else_branch"):
                    sgs[attr.name] = attr.g
            return [sgs["then_branch"], sgs["else_branch"]]
    return [model.graph]


def _convert_bf16_to_fp32(model):
    """Convert all bf16 tensors and type references to fp32 for WebGPU compatibility.

    WebGPU doesn't support bfloat16. This converts:
    - Graph inputs/outputs elem_type from bf16 → fp32
    - Subgraph inputs/outputs elem_type from bf16 → fp32
    - Initializer data_type from bf16 → fp32 (both inline and external)
    - Cast nodes targeting bf16 → target fp32 instead

    Returns a set of initializer names whose external data needs bf16→fp32
    conversion during resharding.
    """
    BF16 = TensorProto.BFLOAT16  # 16
    FP32 = TensorProto.FLOAT     # 1
    BF16_ELEM = 16  # onnx.TensorProto.BFLOAT16 elem_type for TypeProto

    converted_count = 0
    bf16_external_inits = set()  # names of external inits that were bf16

    def _fix_value_info(vi):
        """Fix a single ValueInfoProto's tensor type."""
        nonlocal converted_count
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == BF16_ELEM:
            vi.type.tensor_type.elem_type = FP32
            converted_count += 1

    def _fix_graph(graph):
        """Fix all bf16 references in a graph."""
        nonlocal converted_count
        # Fix inputs/outputs/value_info
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            _fix_value_info(vi)

        # Fix initializers
        for init in graph.initializer:
            if init.data_type == BF16:
                has_external = any(e.key == "location" for e in init.external_data)
                if has_external:
                    # External data — mark for conversion during resharding
                    bf16_external_inits.add(init.name)
                elif init.raw_data:
                    # Inline data — convert in place
                    bf16_arr = np.frombuffer(init.raw_data, dtype=np.uint16)
                    fp32_arr = (bf16_arr.astype(np.uint32) << 16).view(np.float32)
                    init.raw_data = fp32_arr.tobytes()
                init.data_type = FP32
                converted_count += 1

        # Fix nodes
        for node in graph.node:
            # Fix Constant nodes with bf16 tensor values (e.g. RoPE cos/sin)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t and attr.t.data_type == BF16:
                        if attr.t.raw_data:
                            bf16_arr = np.frombuffer(attr.t.raw_data, dtype=np.uint16)
                            fp32_arr = (bf16_arr.astype(np.uint32) << 16).view(np.float32)
                            attr.t.raw_data = fp32_arr.tobytes()
                        attr.t.data_type = FP32
                        converted_count += 1
            # Fix Cast nodes that target bf16
            elif node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == BF16:
                        attr.i = FP32
                        converted_count += 1
            # Recurse into subgraphs (If/Loop/Scan)
            for attr in node.attribute:
                if attr.g and attr.g.ByteSize() > 0:
                    _fix_graph(attr.g)

    _fix_graph(model.graph)
    print(f"  Converted {converted_count} bf16 references to fp32")
    if bf16_external_inits:
        print(f"  {len(bf16_external_inits)} external initializers need data conversion during reshard")
    return bf16_external_inits


def quantize_q4(merged_path):
    """Quantize merged model to Q4 using MatMulNBits.

    Handles both flat graphs and If-node merged graphs (subgraphs).
    """
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

    # Get subgraphs (works for both flat and If-node models)
    subgraphs = _get_subgraphs(model)
    is_merged = len(subgraphs) == 2
    print(f"  Model type: {'If-node merged' if is_merged else 'flat'} ({len(subgraphs)} subgraph(s))")

    # Top-level initializer map
    init_map = {init.name: init for init in model.graph.initializer}

    # Also include subgraph initializers (small scalars etc.)
    sg_init_map = {}
    for sg in subgraphs:
        for init in sg.initializer:
            sg_init_map[init.name] = init

    # Collect MatMul nodes from all subgraphs
    # (subgraph_ref, node, weight_name, weight_idx, activation_input)
    matmul_entries = []
    matmul_weight_names = set()
    for sg in subgraphs:
        for node in sg.node:
            if node.op_type == "MatMul":
                for idx, inp_name in enumerate(node.input):
                    if inp_name in init_map:
                        init = init_map[inp_name]
                        info = get_external_info(init)
                        if ("location" in info and
                                init.data_type in (TensorProto.FLOAT, TensorProto.BFLOAT16) and
                                "embed_tokens" not in inp_name and
                                len(init.dims) == 2):
                            matmul_weight_names.add(inp_name)
                            matmul_entries.append((sg, node, inp_name, idx, node.input[1 - idx]))
                        break

    print(f"  Unique MatMul weights to quantize: {len(matmul_weight_names)}")
    print(f"  MatMul nodes to replace: {len(matmul_entries)}")

    src_data_info = {}
    for init in model.graph.initializer:
        info = get_external_info(init)
        if "location" in info:
            src_data_info[init.name] = info

    # Quantize each unique weight once
    quantized_data = {}  # weight_name → (b_name, scales_name, zp_name, K, N)
    inits_to_add = []
    inits_to_remove = set()

    for weight_name in sorted(matmul_weight_names):
        init = init_map[weight_name]
        info = src_data_info.get(weight_name)
        if info is None:
            continue

        K, N = list(init.dims)

        # Read weight data from external file
        src_file = src_dir / info["location"]
        offset = int(info.get("offset", "0"))
        length = int(info["length"])
        with open(src_file, "rb") as f:
            f.seek(offset)
            raw_data = f.read(length)

        # Handle both FP32 and BF16 data
        if init.data_type == TensorProto.BFLOAT16:
            bf16_arr = np.frombuffer(raw_data, dtype=np.uint16)
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

        b_name = f"{weight_name}_q4"
        scales_name = f"{weight_name}_scales"
        zp_name = f"{weight_name}_zp"

        inits_to_add.append(numpy_helper.from_array(packed, name=b_name))
        inits_to_add.append(numpy_helper.from_array(block_scales, name=scales_name))
        inits_to_add.append(numpy_helper.from_array(zp_packed, name=zp_name))
        inits_to_remove.add(weight_name)

        quantized_data[weight_name] = (b_name, scales_name, zp_name, K, N)

        if len(quantized_data) % 20 == 0:
            print(f"  Quantized {len(quantized_data)} weights...")

    print(f"  Total unique weights quantized: {len(quantized_data)}")

    # Replace MatMul → MatMulNBits in subgraphs
    replace_count = 0
    for sg, node, weight_name, weight_idx, activation_input in matmul_entries:
        if weight_name not in quantized_data:
            continue
        b_name, scales_name, zp_name, K, N = quantized_data[weight_name]

        matmul_nbits = helper.make_node(
            "MatMulNBits",
            inputs=[activation_input, b_name, scales_name, zp_name],
            outputs=node.output,
            name=node.name + "_q4" if node.name else f"matmulnbits_{replace_count}",
            domain="com.microsoft",
            K=K, N=N, bits=4, block_size=BLOCK_SIZE,
        )

        # Replace in-place within the subgraph's node list
        for i, n in enumerate(sg.node):
            if n is node:
                sg.node.remove(n)
                sg.node.insert(i, matmul_nbits)
                break

        replace_count += 1

    print(f"  Replaced {replace_count} MatMul nodes across {len(subgraphs)} subgraph(s)")

    # Update top-level initializers
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
            print(f"  embed_tokens: {len(raw)/1e6:.1f} MB -> FP16 {len(fp16_data)/1e6:.1f} MB")

            orig_name = init.name
            fp16_name = init.name + "_fp16"
            init.name = fp16_name
            init.data_type = TensorProto.FLOAT16
            clear_external_data(init)
            init.raw_data = fp16_data

            # Update Gather nodes in all subgraphs
            for sg in subgraphs:
                for node in sg.node:
                    if node.op_type == "Gather":
                        for idx, inp in enumerate(node.input):
                            if inp == orig_name:
                                node.input[idx] = fp16_name
                                orig_out = node.output[0]
                                node.output[0] = orig_out + "_fp16"
                                cast = helper.make_node(
                                    "Cast", inputs=[orig_out + "_fp16"], outputs=[orig_out],
                                    name=f"cast_embed_fp16_{sg.name}", to=TensorProto.FLOAT,
                                )
                                node_idx = list(sg.node).index(node)
                                sg.node.insert(node_idx + 1, cast)
                                break
            break

    # Convert bf16 → fp32 for WebGPU compatibility (WebGPU doesn't support bf16)
    print("\n  Converting bf16 → fp32 for WebGPU compatibility...")
    bf16_external_inits = _convert_bf16_to_fp32(model)

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

        # External data — convert bf16→fp32 if needed
        src_file = src_dir / info["location"]
        src_offset = int(info.get("offset", "0"))
        src_length = int(info["length"])
        with open(src_file, "rb") as f:
            f.seek(src_offset)
            data = f.read(src_length)
        if init.name in bf16_external_inits:
            bf16_arr = np.frombuffer(data, dtype=np.uint16)
            fp32_arr = (bf16_arr.astype(np.uint32) << 16).view(np.float32)
            data = fp32_arr.tobytes()
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

    # Verify Q4 model retains merged inputs/outputs
    q4 = onnx.load(str(out_path), load_external_data=False)
    q4_inputs = [i.name for i in q4.graph.input]
    q4_outputs = [o.name for o in q4.graph.output]
    print(f"\n  Q4 inputs ({len(q4_inputs)}): {q4_inputs[:3]} ... {q4_inputs[-1]}")
    print(f"  Q4 outputs ({len(q4_outputs)}): {q4_outputs[:2]} ... {q4_outputs[-1]}")
    print(f"  use_cache_branch: {'use_cache_branch' in q4_inputs}")
    print(f"  past_key_values: {any(n.startswith('past_key_values') for n in q4_inputs)}")
    del q4

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
