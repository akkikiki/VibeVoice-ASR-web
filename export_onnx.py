"""
Export VibeVoice-ASR model to ONNX format for use with Transformers.js.

Exports 2 ONNX subgraphs:
  1. Speech Encoder: audio waveform -> speech embeddings
     (acoustic tokenizer + semantic tokenizer + connectors)
  2. Decoder (merged): input_ids/inputs_embeds + KV-cache -> logits + present KV
     Handles both prefill (with speech embeddings) and autoregressive steps.

Usage:
    python export_onnx.py --output_dir ./onnx_output
    python export_onnx.py --output_dir ./onnx_output --no_kv_cache  # legacy mode
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wrapper modules for clean ONNX export
# ---------------------------------------------------------------------------

class SpeechEncoderForONNX(nn.Module):
    """Wraps acoustic + semantic tokenizer encoders and connectors."""

    def __init__(self, asr_model):
        super().__init__()
        self.acoustic_encoder = asr_model.model.acoustic_tokenizer.encoder
        self.semantic_encoder = asr_model.model.semantic_tokenizer.encoder
        self.acoustic_connector = asr_model.model.acoustic_connector
        self.semantic_connector = asr_model.model.semantic_connector

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, 1, samples)
        # Encode through both tokenizers
        ac_latents = self.acoustic_encoder(audio).transpose(1, 2)  # (B, T, 64)
        se_latents = self.semantic_encoder(audio).transpose(1, 2)  # (B, T, 128)

        # Project to LM hidden size and combine
        ac_emb = self.acoustic_connector(ac_latents)  # (B, T, hidden_size)
        se_emb = self.semantic_connector(se_latents)  # (B, T, hidden_size)

        return ac_emb + se_emb  # (B, T, hidden_size)


class DecoderMergedForONNX(nn.Module):
    """Merged decoder that handles both prefill and autoregressive steps.

    This supports Transformers.js's expected interface for encoder-decoder models:
    - Prefill: receives inputs_embeds (speech + text embeddings) with no KV-cache
    - Autoregressive: receives input_ids (single token) with past KV-cache

    The model outputs logits and present KV-cache tensors.
    """

    def __init__(self, asr_model):
        super().__init__()
        self.embed_tokens = asr_model.model.language_model.embed_tokens
        self.language_model = asr_model.model.language_model
        self.lm_head = asr_model.lm_head
        self.config = asr_model.config.decoder_config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = True,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        return logits, outputs.past_key_values


class DecoderForONNX(nn.Module):
    """Wraps Qwen2 language model + lm_head for text generation (no KV-cache)."""

    def __init__(self, asr_model):
        super().__init__()
        self.embed_tokens = asr_model.model.language_model.embed_tokens
        self.language_model = asr_model.model.language_model
        self.lm_head = asr_model.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, seq_len)
        outputs = self.language_model(input_ids=input_ids)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class DecoderWithSpeechForONNX(nn.Module):
    """Decoder that accepts pre-computed speech embeddings concatenated
    before text tokens (first pass / prefill). No KV-cache."""

    def __init__(self, asr_model):
        super().__init__()
        self.embed_tokens = asr_model.model.language_model.embed_tokens
        self.language_model = asr_model.model.language_model
        self.lm_head = asr_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        speech_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids: (B, text_len) - text token ids
        # speech_embeddings: (B, speech_len, hidden_size) - from speech encoder

        text_embeds = self.embed_tokens(input_ids)  # (B, text_len, hidden_size)

        # Concatenate: [speech_embeddings, text_embeddings]
        combined = torch.cat([speech_embeddings, text_embeds], dim=1)

        outputs = self.language_model(inputs_embeds=combined)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_speech_encoder(model, output_dir: str, dtype: torch.dtype):
    print("\n=== Exporting Speech Encoder ===")
    encoder = SpeechEncoderForONNX(model).to(dtype).eval()

    # 1 second of audio at 24kHz
    dummy_audio = torch.randn(1, 1, 24000, dtype=dtype)

    with torch.no_grad():
        out = encoder(dummy_audio)
        print(f"  Test output shape: {out.shape}")

    path = os.path.join(output_dir, "speech_encoder.onnx")
    print(f"  Exporting to {path}...")

    t0 = time.time()
    torch.onnx.export(
        encoder,
        dummy_audio,
        path,
        input_names=["audio"],
        output_names=["speech_embeddings"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "speech_embeddings": {0: "batch", 1: "time"},
        },
        opset_version=18,
    )
    elapsed = time.time() - t0

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Done in {elapsed:.1f}s — {size_mb:.1f} MB")
    return path


def export_decoder_merged(model, output_dir: str, dtype: torch.dtype):
    """Export a merged decoder with KV-cache support for Transformers.js.

    Produces decoder_model_merged.onnx that handles both:
    - Prefill: inputs_embeds (B, seq_len, hidden) -> logits + KV-cache
    - Autoregressive: input_ids (B, 1) + past KV -> logits + updated KV
    """
    config = model.config.decoder_config
    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    hidden_size = config.hidden_size

    print(f"\n=== Exporting Merged Decoder with KV-cache ===")
    print(f"  num_layers={num_layers}, num_kv_heads={num_heads}, head_dim={head_dim}")
    decoder = DecoderMergedForONNX(model).to(dtype).eval()

    # --- Prefill trace ---
    # For the prefill pass, we use inputs_embeds (speech + text combined)
    prefill_len = 18  # speech_len + text_len
    dummy_embeds = torch.randn(1, prefill_len, hidden_size, dtype=dtype)

    with torch.no_grad():
        logits, past_kv = decoder(inputs_embeds=dummy_embeds, use_cache=True)
        print(f"  Prefill test: embeds {dummy_embeds.shape} -> logits {logits.shape}")
        print(f"  KV-cache layers: {len(past_kv)}, shape per layer: k={past_kv[0][0].shape}, v={past_kv[0][1].shape}")

    # --- Build input/output names and dynamic axes for KV-cache ---
    input_names = ["input_ids", "inputs_embeds"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "inputs_embeds": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }

    # Past KV inputs (for autoregressive steps)
    for i in range(num_layers):
        k_name = f"past_key_values.{i}.key"
        v_name = f"past_key_values.{i}.value"
        input_names.extend([k_name, v_name])
        dynamic_axes[k_name] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[v_name] = {0: "batch", 2: "past_seq_len"}

    # Present KV outputs
    for i in range(num_layers):
        k_name = f"present.{i}.key"
        v_name = f"present.{i}.value"
        output_names.extend([k_name, v_name])
        dynamic_axes[k_name] = {0: "batch", 2: "total_seq_len"}
        dynamic_axes[v_name] = {0: "batch", 2: "total_seq_len"}

    # For the ONNX export, we need a wrapper that flattens past_key_values
    class DecoderMergedONNXWrapper(nn.Module):
        def __init__(self, decoder_merged, num_layers):
            super().__init__()
            self.decoder = decoder_merged
            self.num_layers = num_layers

        def forward(self, input_ids, inputs_embeds, *past_kv_flat):
            # Reconstruct past_key_values from flat tensors
            past_key_values = None
            if len(past_kv_flat) > 0 and past_kv_flat[0].shape[2] > 0:
                past_key_values = []
                for i in range(self.num_layers):
                    k = past_kv_flat[2 * i]
                    v = past_kv_flat[2 * i + 1]
                    past_key_values.append((k, v))

            # Determine if this is prefill or decode step
            # During prefill: inputs_embeds has seq_len > 0
            # During decode: input_ids has seq_len > 0
            if inputs_embeds.shape[1] > 0:
                embeds = inputs_embeds
            else:
                embeds = self.decoder.embed_tokens(input_ids)

            logits, present_kv = self.decoder(
                inputs_embeds=embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Flatten present KV for output
            outputs = [logits]
            for k, v in present_kv:
                outputs.extend([k, v])
            return tuple(outputs)

    wrapper = DecoderMergedONNXWrapper(decoder, num_layers).to(dtype).eval()

    # Dummy inputs for export (prefill mode)
    dummy_input_ids = torch.zeros(1, 0, dtype=torch.long)  # empty during prefill
    dummy_inputs_embeds = torch.randn(1, prefill_len, hidden_size, dtype=dtype)

    # Empty past KV (no cache during prefill)
    dummy_past_kv = []
    for _ in range(num_layers):
        dummy_past_kv.append(torch.zeros(1, num_heads, 0, head_dim, dtype=dtype))  # key
        dummy_past_kv.append(torch.zeros(1, num_heads, 0, head_dim, dtype=dtype))  # value

    all_inputs = (dummy_input_ids, dummy_inputs_embeds, *dummy_past_kv)

    path = os.path.join(output_dir, "decoder_model_merged.onnx")
    print(f"  Exporting to {path}...")

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        all_inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    elapsed = time.time() - t0

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Done in {elapsed:.1f}s — {size_mb:.1f} MB")
    return path


def export_decoder(model, output_dir: str, dtype: torch.dtype):
    print("\n=== Exporting Decoder (text-only, no KV-cache, legacy) ===")
    decoder = DecoderForONNX(model).to(dtype).eval()

    dummy_ids = torch.randint(0, 1000, (1, 10))

    with torch.no_grad():
        out = decoder(dummy_ids)
        print(f"  Test output shape: {out.shape}")

    path = os.path.join(output_dir, "decoder.onnx")
    print(f"  Exporting to {path}...")

    t0 = time.time()
    torch.onnx.export(
        decoder,
        dummy_ids,
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=18,
    )
    elapsed = time.time() - t0

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Done in {elapsed:.1f}s — {size_mb:.1f} MB")
    return path


def export_decoder_with_speech(model, output_dir: str, dtype: torch.dtype):
    hidden_size = model.config.decoder_config.hidden_size

    print("\n=== Exporting Decoder with Speech (prefill, no KV-cache, legacy) ===")
    decoder = DecoderWithSpeechForONNX(model).to(dtype).eval()

    dummy_ids = torch.randint(0, 1000, (1, 10))
    dummy_speech = torch.randn(1, 8, hidden_size, dtype=dtype)

    with torch.no_grad():
        out = decoder(dummy_ids, dummy_speech)
        print(f"  Test output shape: {out.shape}")

    path = os.path.join(output_dir, "decoder_with_speech.onnx")
    print(f"  Exporting to {path}...")

    t0 = time.time()
    torch.onnx.export(
        decoder,
        (dummy_ids, dummy_speech),
        path,
        input_names=["input_ids", "speech_embeddings"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "text_len"},
            "speech_embeddings": {0: "batch", 1: "speech_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=18,
    )
    elapsed = time.time() - t0

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Done in {elapsed:.1f}s — {size_mb:.1f} MB")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export VibeVoice-ASR to ONNX")
    parser.add_argument(
        "--model_path",
        default="microsoft/VibeVoice-ASR",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--output_dir",
        default="./onnx_output",
        help="Directory to save ONNX files",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype for export (default: bfloat16 to match original weights)",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Export legacy models without KV-cache (separate decoder + decoder_with_speech)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    print(f"  dtype: {args.dtype}")

    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )

    config = VibeVoiceConfig.from_pretrained(args.model_path)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

    # Export speech encoder (always needed)
    export_speech_encoder(model, args.output_dir, dtype)

    if args.no_kv_cache:
        # Legacy mode: separate prefill and autoregressive decoders without KV-cache
        export_decoder_with_speech(model, args.output_dir, dtype)
        export_decoder(model, args.output_dir, dtype)
    else:
        # Merged decoder with KV-cache support for Transformers.js
        export_decoder_merged(model, args.output_dir, dtype)

    print("\n=== All exports complete! ===")
    print(f"Files saved to: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".onnx"):
            fpath = os.path.join(args.output_dir, f)
            size = os.path.getsize(fpath)
            # Check for external data file
            data_path = fpath + ".data"
            if os.path.exists(data_path):
                size += os.path.getsize(data_path)
            print(f"  {f}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
