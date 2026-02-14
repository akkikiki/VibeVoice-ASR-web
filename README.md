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

## ONNX Weights

Pre-exported weights: [akkikiki/VibeVoice-ASR-onnx](https://huggingface.co/akkikiki/VibeVoice-ASR-onnx)
