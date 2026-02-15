# VibeVoice-ASR-web

Browser-based speech recognition using [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) via Transformers.js v4 + WebGPU.

**Live Demo:** [huggingface.co/spaces/akkikiki/VibeVoice-ASR](https://huggingface.co/spaces/akkikiki/VibeVoice-ASR)

## Features

- **Browser-based inference** with WebGPU acceleration (no server required)
- **Model selector UI** — choose decode mode (with/without KV-cache) and quantization (INT8, Q4)
- **KV-cache support** — merged decoder with If-node conditional routing between prefill and decode paths
- **Configurable generation** — adjustable max tokens, editable prompt template, stop generation button
- **Repetition penalty** — frequency-based penalty + n-gram blocking to prevent degenerate outputs
- **Mobile detection** — warns users that the 7B model requires desktop-class GPU memory
- **Verbose loading status** — shows download progress and WebGPU shader compilation phase

## Architecture

VibeVoice-ASR is a composite speech recognition model:

| Component | Architecture | Purpose |
|-----------|-------------|---------|
| Speech Encoder | Custom ConvNeXt-style VAE + Transformer | Encodes 24kHz audio → speech embeddings (1, T, 3584) |
| Decoder (LM) | Qwen2-7B (28 layers, 28 heads, 4 KV heads) | Autoregressive text generation from speech + text |

## ONNX Models

The app loads two ONNX subgraphs via `WhisperForConditionalGeneration.from_pretrained()`:

| File | Inputs | Outputs |
|------|--------|---------|
| `encoder_model_fp16.onnx` | `audio` (1, 1, samples) | `speech_embeddings` (1, T, 3584) |
| `decoder_model_merged_{dtype}.onnx` | `input_ids`, `speech_embeddings`, [`past_key_values.*`, `use_cache_branch`] | `logits`, [`present.*`] |

### Quantization Options

| DType | Decoder Size | Total Download | Decoder Shards | Notes |
|-------|-------------|---------------|----------------|-------|
| Q4 | ~3 GB | ~5.7 GB | 4 | MatMulNBits (Q4) + int8 DequantizeLinear lm_head |
| INT8 | ~6 GB | ~9 GB | 5 | DequantizeLinear + MatMul (slower shader compilation) |

All external data shards are kept under 1.9 GB for browser `ArrayBuffer` compatibility.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173`, select a model configuration, and click "Load Model". First load downloads the model (~5-9 GB) and compiles WebGPU shaders (may take several minutes for INT8). Subsequent loads use the browser cache.

## Project Structure

```
src/
├── App.tsx                  # Main UI with generation settings
├── worker.js                # Web Worker: model loading, encoding, decoding
├── hooks/useTranscriber.ts  # React hook bridging UI ↔ worker messages
├── components/
│   ├── ModelSelector.tsx    # Decode mode + quantization picker
│   ├── AudioManager.tsx     # Upload / record audio
│   ├── Progress.tsx         # Download / loading progress display
│   └── Transcript.tsx       # Transcription output display
└── utils/
    └── Constants.ts         # Model config, shard counts, prompt template
```

## Export Scripts

Scripts for exporting and quantizing the ONNX models:

| Script | Purpose |
|--------|---------|
| `scripts/export_decoder_with_kvcache.py` | Export merged decoder with KV-cache (If-node routing) |
| `scripts/export_and_merge_q4_kvcache.py` | Q4 quantization + causal mask fixup + bf16→fp32 conversion |
| `scripts/quantize_kvcache_int8.py` | INT8 streaming quantization |
| `scripts/merge_and_quantize_q4.py` | Q4 quantization pipeline |

### WebGPU Compatibility Fixes (applied during export)

- **bf16→fp32 conversion** — all tensors, Cast nodes, and Constant nodes (RoPE cos/sin), since WebGPU doesn't support bfloat16
- **Dynamic causal mask** — replaces hardcoded seq_len constants with Shape→Gather ops for variable-length sequences
- **Rare dummy seq_len (73)** — avoids accidentally replacing unrelated constants during graph fixup
- **Topological sort fix** — ensures fixup nodes are inserted after their dynamic scalar producers

## Decoding Strategy

- **Greedy decoding** with token suppression (`[Unintelligible Speech]` penalty)
- **Frequency-based repetition penalty** (1.3^n per token occurrence)
- **4-gram blocking** — hard suppresses tokens that would create a 4-gram seen 3+ times
- **Configurable max tokens** (default: 128)
- **Stop generation** button for user-initiated interruption

## Known Issues

- **WebGPU shader compilation is slow** — INT8 decoder (~2300 nodes) takes 10+ minutes on first load due to per-op shader compilation. Q4 is faster because `MatMulNBits` is a single fused op vs `DequantizeLinear + MatMul` (2 ops per layer).
- **No EOS emission** — the quantized model sometimes fails to emit `<|im_end|>` (token 151645), causing generation to run until max tokens. This is a quantization quality issue.
- **Mobile not supported** — the 7B model requires more GPU memory than mobile devices can allocate. Desktop with 16+ GB RAM recommended.
- **INT8 requires more GPU memory** — ~9 GB total, may OOM on machines with ≤32 GB unified memory. Q4 (~5.7 GB) is recommended for most setups.
- **Custom encoder architecture** — the speech encoder uses a non-standard ConvNeXt-style VAE. Do NOT apply standard transformer graph optimizations to it (breaks the model).

## Pre-exported Weights

[akkikiki/VibeVoice-ASR-onnx](https://huggingface.co/akkikiki/VibeVoice-ASR-onnx) on HuggingFace
