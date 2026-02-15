# VibeVoice-ASR-web

ONNX export tools for running [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) in the browser via Transformers.js + WebGPU.

## Features

- **Browser-based inference** with WebGPU acceleration
- **Model selector UI** — choose decode mode (with/without KV-cache) and quantization (INT8 ~9 GB, Q4 ~5.4 GB)
- **KV-cache support** — merged decoder model with If-node conditional routing between prefill and decode paths for fast autoregressive generation
- **Q4 quantization** — 4-bit MatMulNBits for small download sizes

## Exported Subgraphs

### Without KV-cache

| Model | Input | Output |
|-------|-------|--------|
| `speech_encoder` | audio waveform | speech embeddings |
| `decoder_with_speech` | input_ids + speech embeddings | logits (prefill) |
| `decoder` | input_ids | logits (autoregressive) |

### With KV-cache (merged decoder)

| Model | Input | Output |
|-------|-------|--------|
| `speech_encoder` | audio waveform | speech embeddings |
| `decoder_model_merged` | input_ids + speech_embeddings + past_key_values + use_cache_branch | logits + present_key_values |

The merged decoder uses an ONNX If-node to route between the prefill path (processes speech embeddings, outputs KV-cache) and the decode path (uses cached KVs for fast token generation). Weights are deduplicated across both branches.

## Usage

```bash
# Export to ONNX (default: fp32, without KV-cache)
python export_onnx.py --output_dir ./onnx_output --dtype float32

# Validate against PyTorch
python validate_onnx.py --onnx_dir ./onnx_output

# Quantize
python quantize_onnx.py

# Test transcription
python test_transcription.py

# Export merged Q4 KV-cache decoder (requires ~30-40 GB RAM)
python scripts/export_and_merge_q4_kvcache.py --output_dir ./onnx_kvcache
```

## Export Results

### fp32 (baseline)

| Component | Size |
|-----------|------|
| `speech_encoder.onnx` + `.data` | 2.9 GB |
| `decoder_with_speech.onnx` + `.data` | 30.5 GB |
| `decoder.onnx` + `.data` | 30.5 GB |

### fp16

| Component | Size | Notes |
|-----------|------|-------|
| `speech_encoder.onnx` + `.data` | 1.4 GB | Manual fp16 cast (see below) |
| `decoder_with_speech.onnx` + `.data` | 15.2 GB | Direct PyTorch fp16 export |
| `decoder.onnx` + `.data` | 15.2 GB | Direct PyTorch fp16 export |

**Speech encoder fp16 fix:** The `onnxruntime.transformers.float16.convert_float_to_float16()`
converter produced an empty graph (0 nodes, 0 initializers) for the speech encoder — likely
due to the VAE architecture (Conv1d + GroupNorm + residual blocks) triggering an edge case.
The fix was to manually cast each fp32 weight tensor to fp16 using numpy, which preserved
all 1911 ONNX nodes.

### int4

| Component | Size | MatMul tensors quantized | Cosine sim vs fp32 |
|-----------|------|--------------------------|---------------------|
| `speech_encoder.onnx` + `.data` | 0.75 GB | 71 | 0.998 |
| `decoder_with_speech.onnx` + `.data` | 6.71 GB | 197 | 0.980 |
| `decoder.onnx` + `.data` | 6.71 GB | 197 | 0.990 |
| **Total** | **~14.2 GB** | | |

**Int4 quantization approach:** The standard `onnxruntime.quantization.quantize_dynamic` with
`QUInt4` failed for two reasons: (1) it doesn't support Conv layers for int4, and (2) loading
the 30GB fp32 models OOM'd on 32GB RAM. The solution uses a streaming approach:

1. Load ONNX model structure WITHOUT external data (just the graph)
2. Read each tensor's data from the external `.data` file one at a time
3. For MatMul weights: quantize to int4 with block-wise symmetric quantization
   (block_size=32) and write packed data to output file
4. For other weights: copy bytes directly from source to output file
5. Replace `MatMul` ops with `MatMulNBits` (onnxruntime `com.microsoft` domain)
6. Small tensors (<4KB) are inlined in the model file for shape inference

This approach keeps memory usage under ~500MB regardless of model size. Key gotchas:
- `MatMulNBits` quantizes along K (contraction dim), storing data as (N, ...) transposed
- Protobuf has a 2GB per-field limit — must use external data for large tensors
- `onnx.save(save_as_external_data=True)` can produce bloated files; use
  `convert_model_to_external_data()` + `onnx.save()` or write the data file manually
- Scalar/tiny tensors may use typed fields (e.g. `int64_data`) instead of `raw_data`

### Q4 with KV-cache (merged decoder)

| Component | Size | Notes |
|-----------|------|-------|
| `speech_encoder.onnx` + `.data` | 0.75 GB | Same as int4 |
| `decoder_model_merged.onnx` + `.data` | ~4.7 GB | Merged prefill+decode, Q4 MatMulNBits |
| **Total** | **~5.4 GB** | |

**KV-cache export pipeline** (`scripts/export_and_merge_q4_kvcache.py`):
1. Load Qwen2 language model weights from VibeVoice-ASR in bf16
2. Export two ONNX subgraphs (prefill + decode-with-past)
3. Merge into single model with If-node conditional routing
4. Quantize MatMul layers to 4-bit (MatMulNBits), keep embeddings at fp16
5. Optionally upload to HuggingFace Hub

**WebGPU compatibility fixes applied during export:**
- bf16→fp32 conversion for all tensors, Cast nodes, and Constant nodes (RoPE cos/sin) since WebGPU doesn't support bfloat16
- Dynamic causal mask: replaces hardcoded seq_len constants with Shape→Gather ops for variable-length sequences
- Uses rare dummy seq_len (73) during tracing to avoid replacing unrelated constants in the graph
- Topological sort fix for dynamic scalar tracing from graph

## ONNX Weights

Pre-exported weights: [akkikiki/VibeVoice-ASR-onnx](https://huggingface.co/akkikiki/VibeVoice-ASR-onnx)
