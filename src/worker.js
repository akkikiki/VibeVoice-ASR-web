/**
 * Web Worker for VibeVoice-ASR inference.
 *
 * Uses @huggingface/transformers v4 for both tokenizer and model loading.
 * WhisperForConditionalGeneration.from_pretrained() handles downloading,
 * progress callbacks, sharding, and WebGPU session creation automatically.
 *
 * Messages from main thread:
 *   { type: "load", config }                  — Initialize and load model
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
import { MODEL_ID, EOS_TOKEN_ID } from "./utils/Constants";
import { SHARD_COUNTS, TOTAL_FILE_SIZES } from "./utils/Constants";

// Disable local model loading — always fetch from HuggingFace Hub
env.allowLocalModels = false;

let tokenizer = null;

// ONNX sessions (extracted from transformers.js model)
let speechEncoderSession = null;
let decoderSession = null;
let device = null; // "webgpu" or "wasm"
let currentDecodeMode = "no-kvcache"; // "no-kvcache" or "kvcache"

// Track download progress across all files
const fileProgress = new Map();

/**
 * Wrapper around transformers.js v4 Seq2Seq loading for VibeVoice's custom architecture.
 */
class VibeVoiceASRModel {
    constructor(model) {
        this.config = model.config;
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
 * Both decode modes use the same decoder_model_merged ONNX file.
 * @param {object} config - { decodeMode: "no-kvcache"|"kvcache", dtype: "int8"|"q4" }
 */
async function loadModel(config) {
    const { decodeMode = "no-kvcache", dtype = "int8" } = config || {};
    const shards = SHARD_COUNTS[dtype];
    const totalFileSize = TOTAL_FILE_SIZES[dtype] || 9_000_000_000;

    if (!shards) {
        self.postMessage({
            type: "error",
            data: { message: `No ONNX files available for dtype: ${dtype}` },
        });
        return;
    }

    currentDecodeMode = decodeMode;

    try {
        device = "webgpu";
        console.log(`[worker] Using device: ${device}, decodeMode: ${decodeMode}, dtype: ${dtype}`);

        self.postMessage({
            type: "device_info",
            data: { device },
        });

        // Load tokenizer from HuggingFace Hub
        console.log(`[worker] Loading tokenizer from "${MODEL_ID}"`);
        tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
        console.log("[worker] Tokenizer loaded");

        fileProgress.clear();
        const doneFiles = new Set();
        let sessionPhase = "";

        // Load model with transformers.js v4
        // Both modes use decoder_model_merged (it supports KV-cache internally)
        console.log(`[worker] Loading model: encoder_model_fp16 + decoder_model_merged_${dtype} from "${MODEL_ID}"`);

        const loadStart = performance.now();
        const model = await VibeVoiceASRModel.from_pretrained(MODEL_ID, {
            device: "webgpu",
            dtype: {
                encoder_model: "fp16",
                decoder_model_merged: dtype,
            },
            use_external_data_format: {
                encoder_model: shards.encoder,
                decoder_model_merged: shards.decoder,
            },
            progress_callback: (p) => {
                if (p.status === "progress" && typeof p.loaded === "number") {
                    fileProgress.set(p.file, p.loaded);
                    const loaded = Array.from(fileProgress.values()).reduce((a, b) => a + b, 0);
                    self.postMessage({
                        type: "download_progress",
                        data: {
                            progress: Math.min(100, (loaded / totalFileSize) * 100),
                            file: "model",
                        },
                    });
                } else if (p.status === "done" && p.file && !doneFiles.has(p.file)) {
                    doneFiles.add(p.file);
                    const elapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
                    console.log(`[worker] Downloaded: ${p.file} (${elapsed}s)`);
                } else if (p.status === "initiate" && p.file) {
                    // Detect when session creation starts for a model file
                    const shortName = p.file.split("/").pop();
                    if (shortName.endsWith(".onnx") && !sessionPhase.includes(shortName)) {
                        sessionPhase = shortName;
                        const elapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
                        console.log(`[worker] Loading ONNX file: ${shortName} (${elapsed}s)`);
                    }
                } else if (p.status === "ready" && p.file) {
                    const shortName = p.file.split("/").pop();
                    const elapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
                    console.log(`[worker] Session ready: ${shortName} (${elapsed}s)`);

                    // Update UI status with which session is being created next
                    if (shortName.includes("encoder")) {
                        self.postMessage({
                            type: "loading_status",
                            data: { message: "Encoder session ready. Creating decoder session..." },
                        });
                    } else {
                        self.postMessage({
                            type: "loading_status",
                            data: { message: "All sessions ready!" },
                        });
                    }
                }

                // Show "Creating WebGPU session" when downloads are complete
                if (p.status === "done") {
                    self.postMessage({
                        type: "loading_status",
                        data: { message: "Creating WebGPU session..." },
                    });
                }
            },
        });

        const totalElapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
        console.log(`[worker] All sessions created in ${totalElapsed}s`);

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
 * Run decoder with given input_ids and speech embeddings.
 * Optionally accepts past key-value tensors for KV-cache mode.
 */
async function runDecoder(speechEmbeddings, inputIds, pastKeyValues) {
    const feeds = {
        input_ids: new OrtTensor(
            "int64",
            new BigInt64Array(inputIds.map(BigInt)),
            [1, inputIds.length],
        ),
        speech_embeddings: speechEmbeddings,
    };

    // If past key values provided, add them to feeds
    if (pastKeyValues) {
        for (const [name, tensor] of Object.entries(pastKeyValues)) {
            feeds[name] = tensor;
        }
    }

    const result = await decoderSession.run(feeds);
    return result;
}

/**
 * Greedy argmax over the last position's logits.
 */
function greedySample(logits) {
    const data = logits.data;
    const vocabSize = logits.dims[2];
    const seqLen = logits.dims[1];

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
 * Extract past key-value tensors from decoder output.
 * The merged decoder outputs "present.*.key" and "present.*.value" tensors
 * which should be fed back as "past_key_values.*.key" / "past_key_values.*.value".
 */
function extractPastKeyValues(decoderOutput) {
    const pastKV = {};
    for (const name of Object.keys(decoderOutput)) {
        if (name.startsWith("present.")) {
            // present.0.decoder.key -> past_key_values.0.decoder.key
            const pastName = name.replace("present.", "past_key_values.");
            pastKV[pastName] = decoderOutput[name];
        }
    }
    return pastKV;
}

/**
 * Transcribe without KV-cache: re-run full decoder each step.
 */
async function transcribeNoKVCache(speechEmbeddings, promptTokenIds) {
    const MAX_TOKENS = 50;
    const generatedTokens = [];
    let currentIds = [...promptTokenIds];

    for (let step = 0; step < MAX_TOKENS; step++) {
        console.log(`[worker] Decode step ${step + 1}/${MAX_TOKENS} (no KV-cache)...`);
        const result = await runDecoder(speechEmbeddings, currentIds, null);
        const nextToken = greedySample(result.logits);

        if (nextToken === EOS_TOKEN_ID) {
            console.log(`[worker] EOS at step ${step + 1}`);
            break;
        }

        generatedTokens.push(nextToken);
        currentIds.push(nextToken);

        const partialText = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
        console.log(`[worker] Token ${step + 1}: ${nextToken} = "${partialText}"`);
        self.postMessage({
            type: "transcription_partial",
            data: { text: partialText },
        });
    }

    return generatedTokens;
}

/**
 * Transcribe with KV-cache: run prefill once, then decode with cached key/values.
 */
async function transcribeWithKVCache(speechEmbeddings, promptTokenIds) {
    const MAX_TOKENS = 50;
    const generatedTokens = [];

    // Step 1: Prefill — run full prompt through decoder to get first token + KV cache
    console.log("[worker] Prefill pass (KV-cache mode)...");
    let result = await runDecoder(speechEmbeddings, promptTokenIds, null);
    let nextToken = greedySample(result.logits);
    let pastKeyValues = extractPastKeyValues(result);

    if (nextToken === EOS_TOKEN_ID) {
        console.log("[worker] EOS at prefill");
        return generatedTokens;
    }

    generatedTokens.push(nextToken);
    let partialText = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
    console.log(`[worker] Prefill token: ${nextToken} = "${partialText}"`);
    self.postMessage({
        type: "transcription_partial",
        data: { text: partialText },
    });

    // Step 2: Decode — pass only new token + past KV cache
    for (let step = 1; step < MAX_TOKENS; step++) {
        console.log(`[worker] Decode step ${step + 1}/${MAX_TOKENS} (KV-cache)...`);

        // Only pass the last generated token (KV cache has the rest)
        result = await runDecoder(speechEmbeddings, [nextToken], pastKeyValues);
        nextToken = greedySample(result.logits);
        pastKeyValues = extractPastKeyValues(result);

        if (nextToken === EOS_TOKEN_ID) {
            console.log(`[worker] EOS at step ${step + 1}`);
            break;
        }

        generatedTokens.push(nextToken);
        partialText = tokenizer.decode(generatedTokens, { skip_special_tokens: true });
        console.log(`[worker] Token ${step + 1}: ${nextToken} = "${partialText}"`);
        self.postMessage({
            type: "transcription_partial",
            data: { text: partialText },
        });
    }

    return generatedTokens;
}

/**
 * Run transcription pipeline.
 */
async function transcribe(audioData) {
    self.postMessage({ type: "transcription_start" });

    const t0 = performance.now();

    try {
        // Step 1: Encode speech
        console.log("[worker] Encoding speech...");
        const speechEmbeddings = await encodeSpeech(audioData);
        console.log("[worker] Speech embeddings shape:", speechEmbeddings.dims);

        // Step 2: Prepare prompt tokens
        const audioDuration = (audioData.length / 24000).toFixed(2);
        const promptText = "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n<|im_start|>user\nThis is a " + audioDuration + " seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n<|im_start|>assistant\n";
        const encoded = tokenizer(promptText, { return_tensors: false });
        const rawIds = encoded.input_ids;
        const promptTokenIds = Array.isArray(rawIds) ? rawIds : Array.from(rawIds.data || rawIds);
        console.log("[worker] Prompt tokens:", promptTokenIds.length);

        // Step 3: Autoregressive decoding
        console.log(`[worker] Decode mode: ${currentDecodeMode}`);
        const generatedTokens = currentDecodeMode === "kvcache"
            ? await transcribeWithKVCache(speechEmbeddings, promptTokenIds)
            : await transcribeNoKVCache(speechEmbeddings, promptTokenIds);

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
    const { type, audio, config } = e.data;

    switch (type) {
        case "load":
            await loadModel(config);
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
