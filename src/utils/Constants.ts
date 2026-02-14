export const SAMPLE_RATE = 24000;
export const MODEL_ID = "akkikiki/VibeVoice-ASR-onnx";
export const DEFAULT_MAX_NEW_TOKENS = 512;
// Special token IDs (Qwen2 tokenizer)
export const BOS_TOKEN_ID = 151643;
export const EOS_TOKEN_ID = 151645;
export const PAD_TOKEN_ID = 151643;

// Model configuration types
export type DecodeMode = "no-kvcache" | "kvcache";
export type DType = "int8" | "fp16" | "q4";

export interface ModelConfig {
    decodeMode: DecodeMode;
    dtype: DType;
}

// Shard counts for decoder_model_merged per dtype
// (both decode modes use the same merged decoder ONNX file)
export const SHARD_COUNTS: Partial<Record<DType, { encoder: number; decoder: number }>> = {
    int8: { encoder: 1, decoder: 5 },
    q4: { encoder: 1, decoder: 3 },
};

// Approximate total download sizes in bytes per dtype (for progress bar)
export const TOTAL_FILE_SIZES: Partial<Record<DType, number>> = {
    int8: 9_000_000_000,
    q4: 5_400_000_000,
};

// Available dtype options
export const AVAILABLE_DTYPES: DType[] = ["int8", "q4"];
