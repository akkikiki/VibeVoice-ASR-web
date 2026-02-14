"""
Validate ONNX exports against PyTorch model outputs.

Compares the speech encoder and decoder outputs between:
  - Original PyTorch model (bf16)
  - Exported ONNX models (via ONNX Runtime)

Usage:
    python validate_onnx.py --onnx_dir ./onnx_output
"""

import argparse
import os
import time

import numpy as np
import torch


def load_pytorch_model(model_path: str, dtype: torch.dtype):
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )

    print(f"Loading PyTorch model from {model_path} ...")
    config = VibeVoiceConfig.from_pretrained(model_path)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path, config=config, torch_dtype=dtype,
    )
    model.eval()
    return model


def validate_speech_encoder(model, onnx_dir: str, dtype: torch.dtype):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Validating Speech Encoder")
    print("=" * 60)

    # Generate a reproducible test audio (1 second at 24kHz)
    torch.manual_seed(42)
    test_audio = torch.randn(1, 1, 24000, dtype=dtype)

    # --- PyTorch forward ---
    print("\n[PyTorch] Running speech encoder...")
    with torch.no_grad():
        ac_latents = model.model.acoustic_tokenizer.encoder(test_audio).transpose(1, 2)
        se_latents = model.model.semantic_tokenizer.encoder(test_audio).transpose(1, 2)
        ac_emb = model.model.acoustic_connector(ac_latents)
        se_emb = model.model.semantic_connector(se_latents)
        pt_output = (ac_emb + se_emb).float().numpy()
    print(f"  Output shape: {pt_output.shape}")
    print(f"  Output range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")
    print(f"  Output mean:  {pt_output.mean():.6f}")

    # --- ONNX Runtime forward ---
    onnx_path = os.path.join(onnx_dir, "speech_encoder.onnx")
    print(f"\n[ONNX] Loading {onnx_path} ...")
    session = ort.InferenceSession(onnx_path)

    print("[ONNX] Running speech encoder...")
    # ONNX Runtime needs float32 input
    audio_np = test_audio.float().numpy()
    t0 = time.time()
    onnx_output = session.run(None, {"audio": audio_np})[0]
    elapsed = time.time() - t0
    print(f"  Output shape: {onnx_output.shape}")
    print(f"  Output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print(f"  Output mean:  {onnx_output.mean():.6f}")
    print(f"  Inference time: {elapsed:.3f}s")

    # --- Compare ---
    diff = np.abs(pt_output - onnx_output)
    print(f"\n[Comparison]")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Cosine similarity:  {cosine_sim(pt_output.flatten(), onnx_output.flatten()):.6f}")

    ok = cosine_sim(pt_output.flatten(), onnx_output.flatten()) > 0.999
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def validate_decoder(model, onnx_dir: str, dtype: torch.dtype):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Validating Decoder (text-only)")
    print("=" * 60)

    torch.manual_seed(42)
    test_ids = torch.randint(0, 1000, (1, 20))

    # --- PyTorch forward ---
    print("\n[PyTorch] Running decoder...")
    with torch.no_grad():
        embed = model.model.language_model.embed_tokens(test_ids)
        out = model.model.language_model(inputs_embeds=embed)
        pt_logits = model.lm_head(out.last_hidden_state).float().numpy()
    print(f"  Logits shape: {pt_logits.shape}")
    print(f"  Logits range: [{pt_logits.min():.4f}, {pt_logits.max():.4f}]")

    # --- ONNX Runtime forward ---
    onnx_path = os.path.join(onnx_dir, "decoder.onnx")
    print(f"\n[ONNX] Loading {onnx_path} ...")
    session = ort.InferenceSession(onnx_path)

    print("[ONNX] Running decoder...")
    t0 = time.time()
    onnx_logits = session.run(None, {"input_ids": test_ids.numpy()})[0]
    elapsed = time.time() - t0
    print(f"  Logits shape: {onnx_logits.shape}")
    print(f"  Logits range: [{onnx_logits.min():.4f}, {onnx_logits.max():.4f}]")
    print(f"  Inference time: {elapsed:.3f}s")

    # --- Compare ---
    diff = np.abs(pt_logits - onnx_logits)
    print(f"\n[Comparison]")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")

    # Compare argmax predictions (most important for generation)
    pt_preds = pt_logits.argmax(axis=-1)
    onnx_preds = onnx_logits.argmax(axis=-1)
    match = (pt_preds == onnx_preds).mean()
    print(f"  Argmax match rate:  {match:.2%}")
    print(f"  Cosine similarity:  {cosine_sim(pt_logits.flatten(), onnx_logits.flatten()):.6f}")

    ok = match > 0.9
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def validate_decoder_with_speech(model, onnx_dir: str, dtype: torch.dtype):
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Validating Decoder with Speech (prefill)")
    print("=" * 60)

    hidden_size = model.config.decoder_config.hidden_size
    torch.manual_seed(42)
    test_ids = torch.randint(0, 1000, (1, 10))
    test_speech = torch.randn(1, 8, hidden_size, dtype=dtype)

    # --- PyTorch forward ---
    print("\n[PyTorch] Running decoder with speech...")
    with torch.no_grad():
        text_emb = model.model.language_model.embed_tokens(test_ids)
        combined = torch.cat([test_speech.to(text_emb.dtype), text_emb], dim=1)
        out = model.model.language_model(inputs_embeds=combined)
        pt_logits = model.lm_head(out.last_hidden_state).float().numpy()
    print(f"  Logits shape: {pt_logits.shape}")

    # --- ONNX Runtime forward ---
    onnx_path = os.path.join(onnx_dir, "decoder_with_speech.onnx")
    print(f"\n[ONNX] Loading {onnx_path} ...")
    session = ort.InferenceSession(onnx_path)

    print("[ONNX] Running decoder with speech...")
    t0 = time.time()
    onnx_logits = session.run(None, {
        "input_ids": test_ids.numpy(),
        "speech_embeddings": test_speech.float().numpy(),
    })[0]
    elapsed = time.time() - t0
    print(f"  Logits shape: {onnx_logits.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    # --- Compare ---
    diff = np.abs(pt_logits - onnx_logits)
    pt_preds = pt_logits.argmax(axis=-1)
    onnx_preds = onnx_logits.argmax(axis=-1)
    match = (pt_preds == onnx_preds).mean()

    print(f"\n[Comparison]")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Argmax match rate:  {match:.2%}")
    print(f"  Cosine similarity:  {cosine_sim(pt_logits.flatten(), onnx_logits.flatten()):.6f}")

    ok = match > 0.9
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="microsoft/VibeVoice-ASR")
    parser.add_argument("--onnx_dir", default="./onnx_output")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    model = load_pytorch_model(args.model_path, dtype)

    results = []
    results.append(("Speech Encoder", validate_speech_encoder(model, args.onnx_dir, dtype)))
    results.append(("Decoder", validate_decoder(model, args.onnx_dir, dtype)))
    results.append(("Decoder+Speech", validate_decoder_with_speech(model, args.onnx_dir, dtype)))

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
