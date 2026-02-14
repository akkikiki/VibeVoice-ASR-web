"""
Export VibeVoice-ASR model to ONNX format for use with Transformers.js.

Exports 2 ONNX subgraphs:
  1. Speech Encoder: audio waveform -> speech embeddings
     (acoustic tokenizer + semantic tokenizer + connectors)
  2. Decoder: input_ids + speech_embeddings -> logits
     (Qwen2 language model + lm_head)

Usage:
    python export_onnx.py --output_dir ./onnx_output
"""

import argparse
import os
import time

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


class DecoderForONNX(nn.Module):
    """Wraps Qwen2 language model + lm_head for text generation."""

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
    before text tokens (first pass / prefill)."""

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


def export_decoder(model, output_dir: str, dtype: torch.dtype):
    print("\n=== Exporting Decoder (text-only, for autoregressive steps) ===")
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

    print("\n=== Exporting Decoder with Speech (prefill pass) ===")
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

    # Export each component
    export_speech_encoder(model, args.output_dir, dtype)
    export_decoder_with_speech(model, args.output_dir, dtype)
    export_decoder(model, args.output_dir, dtype)

    print("\n=== All exports complete! ===")
    print(f"Files saved to: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".onnx"):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            print(f"  {f}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
