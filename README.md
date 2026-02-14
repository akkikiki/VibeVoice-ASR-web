# VibeVoice-ASR-web

ONNX export tools for running [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) in the browser via Transformers.js + WebGPU.

## Exported Subgraphs

| Model | Input | Output |
|-------|-------|--------|
| `speech_encoder` | audio waveform | speech embeddings |
| `decoder_with_speech` | input_ids + speech embeddings | logits (prefill) |
| `decoder` | input_ids | logits (autoregressive) |

## Usage

```bash
# Export to ONNX (default: fp32)
python export_onnx.py --output_dir ./onnx_output --dtype float32

# Validate against PyTorch
python validate_onnx.py --onnx_dir ./onnx_output

# Quantize
python quantize_onnx.py

# Test transcription
python test_transcription.py
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

## ONNX Weights

Pre-exported weights: [akkikiki/VibeVoice-ASR-onnx](https://huggingface.co/akkikiki/VibeVoice-ASR-onnx)
