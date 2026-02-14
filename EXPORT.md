# Exporting VibeVoice-ASR Decoder with KV-Cache (Q4)

## Overview

`scripts/export_and_merge_q4_kvcache.py` exports the Qwen2 language model decoder from VibeVoice-ASR with KV-cache support, merges prefill + decode models, and quantizes to Q4 for browser deployment.

## Pipeline Steps

1. **Load model** — Downloads from `microsoft/VibeVoice-ASR`, remaps weights to Qwen2ForCausalLM in bf16 (~14 GB)
2. **Export ONNX** — Prefill (speech_embeddings + KV-cache output) and decode-with-past (KV-cache I/O)
3. **Merge** — Combines prefill + decode into `decoder_model_merged.onnx` via `optimum.onnx.merge_decoders`
4. **Quantize Q4** — MatMul → MatMulNBits, embed_tokens → FP16, reshards to < 1.9 GB per file
5. **Upload** (optional) — Pushes output to `akkikiki/VibeVoice-ASR-onnx` on HuggingFace

## Requirements

- **RAM:** ~30-40 GB peak (ONNX export holds model + traced graph). Does NOT fit on 32 GB machines — use DGX Spark or similar.
- **Python dependencies:**
  ```
  pip install torch transformers safetensors huggingface_hub onnx optimum numpy
  ```

## Usage

```bash
# Full pipeline
python scripts/export_and_merge_q4_kvcache.py

# Full pipeline + upload to HuggingFace
python scripts/export_and_merge_q4_kvcache.py --upload

# Skip export, reuse existing ONNX files (for re-running merge/quantize only)
python scripts/export_and_merge_q4_kvcache.py --skip-export
```

## Output

Files are written to `/tmp/vibevoice-split/q4_kvcache_pipeline/q4_output/`:

- `decoder_model_merged_q4.onnx` — Model protobuf
- `decoder_model_merged_q4.onnx_data` — External data shard 0
- `decoder_model_merged_q4.onnx_data_1`, `_2`, ... — Additional shards (each < 1.9 GB)

After export, update the shard count in `src/` constants to match the number of output shards.
