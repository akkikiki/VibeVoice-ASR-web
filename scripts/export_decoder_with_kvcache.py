#!/usr/bin/env python3
"""
Export the VibeVoice-ASR decoder (Qwen2 language model) as two ONNX models
with KV-cache support for efficient autoregressive generation.

This produces two decoder variants:
  1. decoder_model.onnx        -- "prefill" pass (no cache, processes full prompt + speech)
  2. decoder_with_past_model.onnx -- "decode" pass (consumes KV-cache, generates one token)

These can later be merged into a single decoder_model_merged.onnx using optimum's merge tool.

Model architecture (microsoft/VibeVoice-ASR):
  - VibeVoiceASRForConditionalGeneration wraps a Qwen2 decoder
  - Qwen2 config: 28 layers, hidden_size=3584, 28 attn heads, 4 KV heads
  - KV-cache shape per layer: [batch, 4, seq_len, 128]  (head_dim = 3584/28 = 128)
  - vocab_size: 152064

The speech_encoder (handled separately) produces speech_embeddings of shape
[batch, speech_len, 3584]. During prefill, these embeddings are concatenated
with text token embeddings before being fed to the Qwen2 transformer layers.

Requirements:
    pip install torch transformers accelerate

Usage:
    python export_decoder_with_kvcache.py
    python export_decoder_with_kvcache.py --dtype bf16
    python export_decoder_with_kvcache.py --dtype bf16 --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model constants (from config.json)
# ---------------------------------------------------------------------------
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 28
NUM_KEY_VALUE_HEADS = 4
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
VOCAB_SIZE = 152064
MODEL_ID = "microsoft/VibeVoice-ASR"


# ---------------------------------------------------------------------------
# Wrapper modules for ONNX export
# ---------------------------------------------------------------------------

class DecoderPrefillWrapper(nn.Module):
    """
    Wrapper for the PREFILL (first) step of autoregressive generation.

    During prefill the decoder receives:
      - input_ids: the full token sequence [batch, seq_len]
      - speech_embeddings: encoder output [batch, speech_len, hidden_size]

    Internally it:
      1. Embeds input_ids via the language model's embed_tokens
      2. Replaces a contiguous span of the token embeddings with speech_embeddings.
         The convention used by VibeVoice-ASR is that the speech span starts at
         position 1 (after the BOS token) and has length = speech_len.
         NOTE: This simplified replacement strategy matches the existing ONNX
         decoder's behavior where speech_embeddings are concatenated into the
         hidden states.  The exact insertion logic may need adjustment if the
         model uses acoustic_input_mask for non-contiguous placement.
      3. Runs the Qwen2 decoder with use_cache=True
      4. Returns logits + KV-cache (present key/value tensors)

    This wrapper exposes a clean interface for torch.onnx.export.
    """

    def __init__(self, model):
        super().__init__()
        # Extract the Qwen2 language model from the VibeVoice wrapper.
        # The attribute name depends on the model implementation; common names
        # include 'language_model', 'decoder', or 'model'.
        self.language_model = model.language_model
        self.embed_tokens = self.language_model.model.embed_tokens

    def forward(self, input_ids, speech_embeddings):
        """
        Args:
            input_ids: [batch, seq_len] -- token IDs for the full prompt
            speech_embeddings: [batch, speech_len, 3584] -- from speech encoder

        Returns:
            logits: [batch, seq_len, vocab_size]
            present_key_values: tuple of (key, value) for each layer
                each tensor: [batch, num_kv_heads, seq_len, head_dim]
        """
        batch_size, seq_len = input_ids.shape
        speech_len = speech_embeddings.shape[1]

        # Step 1: Embed all input tokens
        inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]

        # Step 2: Replace positions [1 : 1+speech_len] with speech embeddings.
        # This mirrors the model's prepare_inputs_for_generation / forward logic
        # where speech_tensors are inserted at the acoustic_input_mask positions.
        # In the standard setup, the prompt looks like:
        #   [BOS] [speech_placeholder_tokens...] [text_tokens...]
        # and the speech_embeddings replace the placeholder region.
        inputs_embeds = torch.cat([
            inputs_embeds[:, :1, :],           # BOS embedding
            speech_embeddings,                  # speech encoder output
            inputs_embeds[:, 1 + speech_len:, :],  # remaining text tokens
        ], dim=1)

        # Step 3: Build a causal attention mask and position IDs
        # Position IDs must match actual sequence length after speech insertion,
        # NOT the original input_ids length (which may be shorter than speech_len).
        actual_seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(actual_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Step 4: Run the Qwen2 decoder
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Step 5: Extract KV-cache tensors.
        # outputs.past_key_values is a tuple of (key, value) per layer.
        # For DynamicCache (transformers >= 4.36), we access key_cache / value_cache.
        past_kv = outputs.past_key_values
        present_tensors = []
        for i in range(NUM_HIDDEN_LAYERS):
            if hasattr(past_kv, 'key_cache'):
                # DynamicCache format (transformers >= 4.36)
                present_tensors.append(past_kv.key_cache[i])
                present_tensors.append(past_kv.value_cache[i])
            else:
                # Legacy tuple format
                present_tensors.append(past_kv[i][0])
                present_tensors.append(past_kv[i][1])

        return (logits, *present_tensors)


class DecoderWithPastWrapper(nn.Module):
    """
    Wrapper for the DECODE (subsequent) steps of autoregressive generation.

    After prefill, each new token is generated one at a time using the
    KV-cache from previous steps. No speech inputs are needed here --
    the speech information is already captured in the KV-cache.

    Inputs:
      - input_ids: [batch, 1] -- the single newly generated token
      - past_key_values: 56 tensors (28 layers x 2 for key+value)
          each: [batch, num_kv_heads, past_len, head_dim]

    Outputs:
      - logits: [batch, 1, vocab_size]
      - present_key_values: 56 tensors with updated cache
          each: [batch, num_kv_heads, past_len+1, head_dim]
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.language_model
        self.embed_tokens = self.language_model.model.embed_tokens

    def forward(self, input_ids, *past_key_values_flat):
        """
        Args:
            input_ids: [batch, 1]
            *past_key_values_flat: 56 tensors, alternating key/value per layer
                past_key_values_flat[2*i]   = key  for layer i
                past_key_values_flat[2*i+1] = value for layer i

        Returns:
            logits: [batch, 1, vocab_size]
            present_key_values: 56 tensors (updated cache)
        """
        batch_size = input_ids.shape[0]

        # Reconstruct the past_key_values structure expected by Qwen2.
        # We use DynamicCache if available, otherwise fall back to tuple format.
        try:
            from transformers.cache_utils import DynamicCache
            past_kv = DynamicCache()
            for i in range(NUM_HIDDEN_LAYERS):
                key = past_key_values_flat[2 * i]
                value = past_key_values_flat[2 * i + 1]
                # DynamicCache.update() appends to existing cache, but we want
                # to set the full cache. We directly assign to the internal lists.
                past_kv.key_cache.append(key)
                past_kv.value_cache.append(value)
            # Set the seen tokens count so position_ids are computed correctly
            past_kv._seen_tokens = past_key_values_flat[0].shape[2]
        except ImportError:
            # Fall back to legacy tuple format
            past_kv = tuple(
                (past_key_values_flat[2 * i], past_key_values_flat[2 * i + 1])
                for i in range(NUM_HIDDEN_LAYERS)
            )

        # Compute position IDs: the new token position = past sequence length
        past_len = past_key_values_flat[0].shape[2]  # [batch, heads, past_len, head_dim]
        position_ids = torch.tensor([[past_len]], device=input_ids.device).expand(batch_size, -1)

        # Embed the single new token
        inputs_embeds = self.embed_tokens(input_ids)  # [batch, 1, hidden_size]

        # Run decoder with cache
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits  # [batch, 1, vocab_size]

        # Extract updated KV-cache
        new_past_kv = outputs.past_key_values
        present_tensors = []
        for i in range(NUM_HIDDEN_LAYERS):
            if hasattr(new_past_kv, 'key_cache'):
                present_tensors.append(new_past_kv.key_cache[i])
                present_tensors.append(new_past_kv.value_cache[i])
            else:
                present_tensors.append(new_past_kv[i][0])
                present_tensors.append(new_past_kv[i][1])

        return (logits, *present_tensors)


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def build_prefill_io_names():
    """Build input/output names and dynamic axes for the prefill model."""
    input_names = ["input_ids", "speech_embeddings"]
    output_names = ["logits"]

    # Add present KV-cache output names: present.0.key, present.0.value, ...
    for i in range(NUM_HIDDEN_LAYERS):
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    # Dynamic axes: dimensions that can vary at runtime
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "speech_embeddings": {0: "batch_size", 1: "speech_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }
    for i in range(NUM_HIDDEN_LAYERS):
        # present KV shape: [batch, num_kv_heads, seq_len, head_dim]
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "seq_len"}

    return input_names, output_names, dynamic_axes


def build_decode_io_names():
    """Build input/output names and dynamic axes for the decode-with-past model."""
    input_names = ["input_ids"]
    output_names = ["logits"]

    # Add past KV-cache input names and present KV-cache output names
    for i in range(NUM_HIDDEN_LAYERS):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
    for i in range(NUM_HIDDEN_LAYERS):
        # past KV: [batch, num_kv_heads, past_len, head_dim]
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_len"}
        # present KV: [batch, num_kv_heads, past_len+1, head_dim]
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "total_len"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "total_len"}

    return input_names, output_names, dynamic_axes


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def load_model(dtype_str: str, device: str):
    """
    Load the VibeVoice-ASR model from HuggingFace Hub.

    Args:
        dtype_str: "bf16" or "fp32"
        device: "cpu" or "cuda"

    Returns:
        The loaded model instance in fp32 (for correct ONNX export).
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    # Always load in bf16 first to save memory (~14 GB), then convert to fp32
    # for ONNX export. Direct fp32 loading would need ~28 GB.
    load_dtype = torch.bfloat16

    print(f"Loading model {MODEL_ID} with dtype=bf16 on device={device}...")
    print("  (Loading in bf16 first, then converting to fp32 for export)")

    # VibeVoice-ASR uses custom code, so trust_remote_code is required.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=load_dtype,
        trust_remote_code=True,
        device_map=device,
    )

    if dtype_str == "fp32":
        print("  Converting model to fp32...")
        model = model.float()

    model.eval()
    print(f"Model loaded successfully. Type: {type(model).__name__}")
    return model


def export_prefill(model, output_dir: Path, device: str, export_dtype: torch.dtype):
    """
    Export the prefill decoder model (no KV-cache input).

    This model takes the full input sequence + speech embeddings and produces
    logits + initial KV-cache.
    """
    print("\n" + "=" * 70)
    print("Exporting PREFILL decoder (decoder_model.onnx)")
    print("=" * 70)

    wrapper = DecoderPrefillWrapper(model)
    wrapper.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    seq_len = 10       # total tokens (BOS + speech_placeholders + text)
    speech_len = 5     # number of speech embedding frames

    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)
    dummy_speech_embeddings = torch.randn(
        batch_size, speech_len, HIDDEN_SIZE, dtype=export_dtype, device=device
    )

    input_names, output_names, dynamic_axes = build_prefill_io_names()
    output_path = output_dir / "decoder_model.onnx"

    print(f"  Inputs:  {input_names}")
    print(f"  Outputs: logits + {NUM_HIDDEN_LAYERS * 2} present KV tensors")
    print(f"  Saving to: {output_path}")

    # Run a forward pass first to verify the wrapper works
    print("  Running test forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, dummy_speech_embeddings)
    print(f"  Test pass OK. logits shape: {test_out[0].shape}")
    print(f"  KV-cache[0] key shape: {test_out[1].shape}")

    # Export to ONNX
    print("  Exporting to ONNX (this may take several minutes)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_speech_embeddings),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def export_decode_with_past(model, output_dir: Path, device: str, export_dtype: torch.dtype):
    """
    Export the decode-with-past decoder model (with KV-cache input).

    This model takes a single new token + past KV-cache and produces
    logits + updated KV-cache.
    """
    print("\n" + "=" * 70)
    print("Exporting DECODE-WITH-PAST decoder (decoder_with_past_model.onnx)")
    print("=" * 70)

    wrapper = DecoderWithPastWrapper(model)
    wrapper.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    past_len = 10  # simulate 10 tokens already processed

    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, 1), device=device)

    # Create dummy past_key_values: 56 tensors (28 layers x key + value)
    # Each has shape [batch, num_kv_heads, past_len, head_dim]
    dummy_past_kv = []
    for _ in range(NUM_HIDDEN_LAYERS):
        dummy_past_kv.append(
            torch.randn(batch_size, NUM_KEY_VALUE_HEADS, past_len, HEAD_DIM,
                        dtype=export_dtype, device=device)
        )  # key
        dummy_past_kv.append(
            torch.randn(batch_size, NUM_KEY_VALUE_HEADS, past_len, HEAD_DIM,
                        dtype=export_dtype, device=device)
        )  # value

    input_names, output_names, dynamic_axes = build_decode_io_names()
    output_path = output_dir / "decoder_with_past_model.onnx"

    print(f"  Inputs:  input_ids + {NUM_HIDDEN_LAYERS * 2} past KV tensors")
    print(f"  Outputs: logits + {NUM_HIDDEN_LAYERS * 2} present KV tensors")
    print(f"  Saving to: {output_path}")

    # Run a forward pass first to verify the wrapper works
    print("  Running test forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, *dummy_past_kv)
    print(f"  Test pass OK. logits shape: {test_out[0].shape}")
    print(f"  KV-cache[0] key shape: {test_out[1].shape}")

    # Export to ONNX
    print("  Exporting to ONNX (this may take several minutes)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, *dummy_past_kv),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  Exported: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export VibeVoice-ASR decoder with KV-cache support to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export in bf16 (recommended, saves memory):
    python export_decoder_with_kvcache.py --dtype bf16

    # Export in fp32 (requires ~28 GB RAM):
    python export_decoder_with_kvcache.py --dtype fp32

    # Export to custom directory:
    python export_decoder_with_kvcache.py --dtype bf16 --output-dir ./my_export/
""",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Data type for model loading. bf16 recommended to fit in ~14 GB. (default: bf16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/vibevoice-split/kvcache_export",
        help="Directory to save exported ONNX models. (default: /tmp/vibevoice-split/kvcache_export/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model loading and export. (default: cpu)",
    )
    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = args.device

    print("=" * 70)
    print("VibeVoice-ASR Decoder KV-Cache ONNX Export")
    print("=" * 70)
    print(f"  Model:      {MODEL_ID}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Device:     {device}")
    print(f"  Output dir: {output_dir}")
    print(f"  Layers:     {NUM_HIDDEN_LAYERS}")
    print(f"  KV heads:   {NUM_KEY_VALUE_HEADS}")
    print(f"  Head dim:   {HEAD_DIM}")
    print(f"  KV shape:   [batch, {NUM_KEY_VALUE_HEADS}, seq_len, {HEAD_DIM}]")
    print()

    # Step 1: Load model
    model = load_model(args.dtype, device)

    # Step 2: Export prefill decoder (no cache)
    prefill_path = export_prefill(model, output_dir, device, export_dtype)

    # Step 3: Export decode-with-past decoder (with cache)
    decode_path = export_decode_with_past(model, output_dir, device, export_dtype)

    # Summary
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print(f"\nExported files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    print("\nNext steps:")
    print("  1. (Optional) Merge into a single model with optimum:")
    print("     from optimum.onnxruntime import ORTModelForCausalLM")
    print(f"     # or use: optimum-cli onnxruntime merge --input {output_dir}")
    print("  2. (Optional) Quantize with onnxruntime or convert embed_tokens to fp16")
    print("  3. (Optional) Reshard external data for browser deployment (<2 GB files)")
    print()


if __name__ == "__main__":
    main()
