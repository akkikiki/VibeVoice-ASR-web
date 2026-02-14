/**
 * Web Worker for VibeVoice-ASR inference.
 *
 * Uses @huggingface/transformers v4 for both tokenizer and model loading.
 * WhisperForConditionalGeneration.from_pretrained() handles downloading,
 * progress callbacks, sharding, and WebGPU session creation automatically.
 *
 * Messages from main thread:
 *   { type: "load" }                         — Initialize and load model
 *   { type: "transcribe", audio, sampleRate } — Transcribe audio
 *
 * Messages to main thread:
 *   { type: "download_progress", data: { progress, file } }
 *   { type: "device_info", data: { device } }
 *   { type: "ready" }
 *   { type: "transcription_start" }
 *   { type: "transcription_partial", data: { text } }
 *   { type: "transcription_complete", data: { text, inferenceTime } }
 *   { type: "error", data: { message } }
 */

import { env, AutoTokenizer, WhisperForConditionalGeneration } from "@huggingface/transformers";
import { Tensor as OrtTensor } from "onnxruntime-web/webgpu";
import { MODEL_ID, EOS_TOKEN_ID, DEFAULT_MAX_NEW_TOKENS } from "./utils/Constants";

// Disable local model loading — always fetch from HuggingFace Hub
env.allowLocalModels = false;

let tokenizer = null;

// ONNX sessions (extracted from transformers.js model)
let speechEncoderSession = null;
let decoderSession = null;
let device = null; // "webgpu" or "wasm"

// Track download progress across all files (like GPT-OSS-WebGPU pattern)
const fileProgress = new Map();
// Total expected download size in bytes (speech_encoder_fp16 ~1.36 GB + decoder_q4 ~5.4 GB)
const TOTAL_FILE_SIZE = 6_700_000_000;

/**
 * Wrapper around transformers.js v4 Seq2Seq loading for VibeVoice's custom architecture.
 * Delegates to WhisperForConditionalGeneration which has the right Seq2Seq session pattern
 * (encoder + decoder_model_merged), then exposes sessions with VibeVoice-specific names.
 */
class VibeVoiceASRModel {
    constructor(model) {
        this.config = model.config;
        // WhisperForConditionalGeneration loads:
        //   sessions.model          -> encoder_model_q4.onnx
        //   sessions.decoder_model_merged -> decoder_model_merged_q4.onnx
        this.speechEncoderSession = model.sessions.model;
        this.decoderSession = model.sessions.decoder_model_merged;
    }

    static async from_pretrained(modelId, options) {
        const model = await WhisperForConditionalGeneration.from_pretrained(modelId, options);
        return new VibeVoiceASRModel(model);
    }
}

/**
 * Load model using transformers.js v4's from_pretrained().
 * Handles downloading, progress, sharding, and WebGPU session creation automatically.
 */
async function loadModel() {
    try {
        device = "webgpu";
        console.log(`[worker] Using device: ${device}`);

        self.postMessage({
            type: "device_info",
            data: { device },
        });

        // Load tokenizer from HuggingFace Hub
        console.log(`[worker] Loading tokenizer from ${MODEL_ID}...`);
        tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
        console.log("[worker] Tokenizer loaded");

        // Load model with transformers.js v4
        // This handles downloading, caching, sharding, and WebGPU session creation
        console.log(`[worker] Loading model from ${MODEL_ID}...`);
        const model = await VibeVoiceASRModel.from_pretrained(MODEL_ID, {
            device: "webgpu",
            dtype: {
                encoder_model: "fp16",
                decoder_model_merged: "q4",
            },
            use_external_data_format: {
                encoder_model: 1,
                decoder_model_merged: 3,
            },
            progress_callback: (p) => {
                if (p.status === "progress" && typeof p.loaded === "number") {
                    fileProgress.set(p.file, p.loaded);
                    const loaded = Array.from(fileProgress.values()).reduce((a, b) => a + b, 0);
                    self.postMessage({
                        type: "download_progress",
                        data: {
                            progress: Math.min(100, (loaded / TOTAL_FILE_SIZE) * 100),
                            file: "model",
                        },
                    });
                } else if (p.status === "done") {
                    // File download complete — after all files, session creation begins
                    self.postMessage({
                        type: "loading_status",
                        data: { message: "Creating WebGPU session..." },
                    });
                }
            },
        });

        speechEncoderSession = model.speechEncoderSession;
        decoderSession = model.decoderSession;

        console.log("[worker] Speech encoder ready");
        console.log("[worker] Encoder inputs:", speechEncoderSession.inputNames);
        console.log("[worker] Encoder outputs:", speechEncoderSession.outputNames);
        console.log("[worker] Decoder ready");
        console.log("[worker] Decoder inputs:", JSON.stringify(decoderSession.inputNames));
        console.log("[worker] Decoder outputs:", JSON.stringify(decoderSession.outputNames));

        self.postMessage({ type: "ready" });
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[worker] Model loading failed:", err);
        console.error("[worker] Error stack:", err?.stack);

        self.postMessage({
            type: "error",
            data: {
                message: `WebGPU model loading failed: ${msg}. ` +
                    `Try using a browser with WebGPU support (Chrome 113+).`,
            },
        });
    }
}

/**
 * Run the speech encoder on raw audio.
 * @param {Float32Array} audioData - Raw audio at 24kHz
 * @returns speech_embeddings tensor (1, T, hidden_size)
 */
async function encodeSpeech(audioData) {
    const audioTensor = new OrtTensor("float32", audioData, [
        1,
        1,
        audioData.length,
    ]);

    const result = await speechEncoderSession.run({ audio: audioTensor });
    return result.speech_embeddings;
}

/**
 * Run prefill: pass input_ids + speech embeddings through decoder.
 * Uses the decoder_model_merged (no KV-cache).
 */
async function runPrefill(speechEmbeddings, promptTokenIds) {
    const feeds = {
        input_ids: new OrtTensor(
            "int64",
            new BigInt64Array(promptTokenIds.map(BigInt)),
            [1, promptTokenIds.length],
        ),
        speech_embeddings: speechEmbeddings,
    };

    const result = await decoderSession.run(feeds);
    return { logits: result.logits };
}

/**
 * Greedy argmax over the last position's logits.
 */
function greedySample(logits) {
    const data = logits.data;
    const vocabSize = logits.dims[2];
    const seqLen = logits.dims[1];

    // Get logits for the last position
    const offset = (seqLen - 1) * vocabSize;
    let maxVal = -Infinity;
    let maxIdx = 0;

    for (let i = 0; i < vocabSize; i++) {
        if (data[offset + i] > maxVal) {
            maxVal = data[offset + i];
            maxIdx = i;
        }
    }

    return maxIdx;
}

/**
 * Run transcription pipeline.
 *
 * Note: Without KV-cache, we can only do the prefill pass and get the
 * first predicted token. Full autoregressive generation requires either
 * the merged decoder with KV-cache or re-running the full model each step.
 */
async function transcribe(audioData) {
    self.postMessage({ type: "transcription_start" });

    const t0 = performance.now();

    try {
        // Step 1: Encode speech
        console.log("[worker] Encoding speech...");
        const speechEmbeddings = await encodeSpeech(audioData);
        console.log("[worker] Speech embeddings shape:", speechEmbeddings.dims);

        // Step 2: Prepare prompt tokens (matching VibeVoice-ASR's expected format)
        const audioDuration = (audioData.length / 24000).toFixed(2);
        const promptText = "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n<|im_start|>user\nThis is a " + audioDuration + " seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n<|im_start|>assistant\n";
        const encoded = tokenizer(promptText, { return_tensors: false });
        // Ensure we have a plain number array (transformers.js v4 may return Tensor-like objects)
        const rawIds = encoded.input_ids;
        const promptTokenIds = Array.isArray(rawIds) ? rawIds : Array.from(rawIds.data || rawIds);
        console.log("[worker] Prompt tokens:", promptTokenIds.length);

        // Step 3: Autoregressive decoding (without KV-cache, re-run full decoder each step)
        const MAX_TOKENS = 50;
        const generatedTokens = [];
        let currentIds = [...promptTokenIds];

        for (let step = 0; step < MAX_TOKENS; step++) {
            console.log(`[worker] Decode step ${step + 1}/${MAX_TOKENS}...`);
            const result = await runPrefill(speechEmbeddings, currentIds);
            const nextToken = greedySample(result.logits);

            // Stop on EOS
            if (nextToken === EOS_TOKEN_ID) {
                console.log(`[worker] EOS at step ${step + 1}`);
                break;
            }

            generatedTokens.push(nextToken);
            currentIds.push(nextToken);

            // Send partial result
            const partialText = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
            console.log(`[worker] Token ${step + 1}: ${nextToken} = "${partialText}"`);
            self.postMessage({
                type: "transcription_partial",
                data: { text: partialText },
            });
        }

        const inferenceTime = performance.now() - t0;
        const text = tokenizer.decode(generatedTokens, { skip_special_tokens: true });

        self.postMessage({
            type: "transcription_complete",
            data: { text, inferenceTime },
        });
    } catch (err) {
        console.error("[worker] Transcription failed:", err);
        self.postMessage({
            type: "error",
            data: { message: `Transcription failed: ${err.message}` },
        });
    }
}

// Message handler
self.onmessage = async (e) => {
    const { type, audio } = e.data;

    switch (type) {
        case "load":
            await loadModel();
            break;

        case "transcribe":
            if (!speechEncoderSession || !decoderSession) {
                self.postMessage({
                    type: "error",
                    data: { message: "Model not loaded yet" },
                });
                return;
            }
            await transcribe(audio);
            break;
    }
};
