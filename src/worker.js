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
import { MODEL_ID, EOS_TOKEN_ID, DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT_TEMPLATE } from "./utils/Constants";
import { SHARD_COUNTS, TOTAL_FILE_SIZES } from "./utils/Constants";

// Disable local model loading — always fetch from HuggingFace Hub
env.allowLocalModels = false;

let tokenizer = null;

// ONNX sessions (extracted from transformers.js model)
let speechEncoderSession = null;
let decoderSession = null;
let device = null; // "webgpu" or "wasm"
let currentDecodeMode = "no-kvcache"; // "no-kvcache" or "kvcache"
let maxTokens = DEFAULT_MAX_NEW_TOKENS;
let stopRequested = false;

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
        let allDownloaded = false;
        let sessionStartTime = null;

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
                    const shortName = p.file.split("/").pop();
                    if (shortName.endsWith(".onnx")) {
                        const elapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
                        console.log(`[worker] Loading ONNX file: ${shortName} (${elapsed}s)`);

                        // Once downloads are done, show session creation status
                        if (allDownloaded) {
                            sessionStartTime = performance.now();
                            const modelName = shortName.includes("encoder") ? "encoder" : "decoder";
                            self.postMessage({
                                type: "loading_status",
                                data: { message: `Creating WebGPU session for ${modelName}... (compiling shaders)` },
                            });
                        }
                    }
                } else if (p.status === "ready" && p.file) {
                    const shortName = p.file.split("/").pop();
                    const elapsed = ((performance.now() - loadStart) / 1000).toFixed(1);
                    const sessionTime = sessionStartTime
                        ? ((performance.now() - sessionStartTime) / 1000).toFixed(1)
                        : "?";
                    console.log(`[worker] Session ready: ${shortName} (${elapsed}s total, ${sessionTime}s compile)`);

                    if (shortName.includes("encoder")) {
                        sessionStartTime = performance.now();
                        self.postMessage({
                            type: "loading_status",
                            data: { message: `Encoder ready (${sessionTime}s). Creating decoder session... (compiling shaders, this may take a few minutes)` },
                        });
                    } else {
                        self.postMessage({
                            type: "loading_status",
                            data: { message: `All sessions ready! (decoder compiled in ${sessionTime}s)` },
                        });
                    }
                }

                // Transition from download phase to session creation phase
                if (p.status === "done" && !allDownloaded) {
                    const loaded = Array.from(fileProgress.values()).reduce((a, b) => a + b, 0);
                    if (loaded >= totalFileSize * 0.95) {
                        allDownloaded = true;
                        console.log("[worker] All files downloaded, creating WebGPU sessions...");
                        self.postMessage({
                            type: "loading_status",
                            data: { message: "Downloads complete. Creating WebGPU sessions... (compiling shaders)" },
                        });
                    }
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

// KV-cache config: 28 layers, 4 KV heads, 128 head_dim
const NUM_LAYERS = 28;
const NUM_KV_HEADS = 4;
const HEAD_DIM = 128;

/**
 * Check if the decoder session has use_cache_branch input (merged KV-cache model).
 */
function hasKVCacheInputs() {
    return decoderSession && decoderSession.inputNames.includes("use_cache_branch");
}

/**
 * Run decoder with given input_ids and speech embeddings.
 * For merged KV-cache models, also passes use_cache_branch and past_key_values.
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

    // For merged KV-cache models: always provide use_cache_branch + past_key_values
    if (hasKVCacheInputs()) {
        const useCacheBranch = !!pastKeyValues;
        feeds["use_cache_branch"] = new OrtTensor("bool", [useCacheBranch], [1]);

        if (pastKeyValues) {
            // Decode step: use actual KV-cache from previous step
            for (const [name, tensor] of Object.entries(pastKeyValues)) {
                feeds[name] = tensor;
            }
        } else {
            // Prefill step: provide zero-length KV-cache tensors
            for (let i = 0; i < NUM_LAYERS; i++) {
                const emptyKV = new OrtTensor(
                    "float32",
                    new Float32Array(0),
                    [1, NUM_KV_HEADS, 0, HEAD_DIM],
                );
                feeds[`past_key_values.${i}.key`] = emptyKV;
                feeds[`past_key_values.${i}.value`] = emptyKV;
            }
        }
    } else if (pastKeyValues) {
        // Legacy non-merged model with KV-cache outputs only
        for (const [name, tensor] of Object.entries(pastKeyValues)) {
            feeds[name] = tensor;
        }
    }

    const result = await decoderSession.run(feeds);
    return result;
}

// Token IDs to suppress during decoding (reduce hallucinated "[Unintelligible Speech]")
// "[Unintelligible Speech]" = [58, 1806, 396, 6703, 1238, 38741, 60]
// Penalize "Un" (1806) which starts "Unintelligible" — the main trigger token
const SUPPRESSED_TOKENS = [1806];
const SUPPRESSION_PENALTY = 10.0; // subtracted from logit

/**
 * Greedy argmax over the last position's logits, with optional token suppression.
 */
function greedySample(logits) {
    const data = logits.data;
    const vocabSize = logits.dims[2];
    const seqLen = logits.dims[1];

    const offset = (seqLen - 1) * vocabSize;
    let maxVal = -Infinity;
    let maxIdx = 0;

    for (let i = 0; i < vocabSize; i++) {
        let score = data[offset + i];
        if (SUPPRESSED_TOKENS.includes(i)) {
            score -= SUPPRESSION_PENALTY;
        }
        if (score > maxVal) {
            maxVal = score;
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
    const generatedTokens = [];
    let currentIds = [...promptTokenIds];

    for (let step = 0; step < maxTokens; step++) {
        if (stopRequested) {
            console.log(`[worker] Stop requested at step ${step + 1}`);
            break;
        }
        console.log(`[worker] Decode step ${step + 1}/${maxTokens} (no KV-cache)...`);
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
    for (let step = 1; step < maxTokens; step++) {
        if (stopRequested) {
            console.log(`[worker] Stop requested at step ${step + 1}`);
            break;
        }
        console.log(`[worker] Decode step ${step + 1}/${maxTokens} (KV-cache)...`);

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
async function transcribe(audioData, promptTemplate, tokenLimit) {
    stopRequested = false;
    maxTokens = tokenLimit || DEFAULT_MAX_NEW_TOKENS;
    self.postMessage({ type: "transcription_start" });

    const t0 = performance.now();

    try {
        // Step 1: Encode speech
        console.log("[worker] Encoding speech...");
        const speechEmbeddings = await encodeSpeech(audioData);
        console.log("[worker] Speech embeddings shape:", speechEmbeddings.dims);

        // Step 2: Prepare prompt tokens
        const audioDuration = (audioData.length / 24000).toFixed(2);
        const promptText = (promptTemplate || DEFAULT_PROMPT_TEMPLATE).replace(/\{duration\}/g, audioDuration);
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
    const { type, audio, config, promptTemplate, maxTokens: msgMaxTokens } = e.data;

    switch (type) {
        case "load":
            await loadModel(config);
            break;

        case "stop":
            stopRequested = true;
            break;

        case "transcribe":
            if (!speechEncoderSession || !decoderSession) {
                self.postMessage({
                    type: "error",
                    data: { message: "Model not loaded yet" },
                });
                return;
            }
            await transcribe(audio, promptTemplate, msgMaxTokens);
            break;
    }
};
