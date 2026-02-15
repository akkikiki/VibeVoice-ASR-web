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

        // Log WebGPU device limits for debugging
        if (navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const adapterInfo = adapter.info || {};
                    const limits = adapter.limits;
                    console.log(`[worker] WebGPU adapter: ${adapterInfo.vendor || "unknown"} / ${adapterInfo.architecture || "unknown"} / ${adapterInfo.device || "unknown"}`);
                    console.log(`[worker] WebGPU maxBufferSize: ${(limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`);
                    console.log(`[worker] WebGPU maxStorageBufferBindingSize: ${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`);
                }
            } catch (e) {
                console.warn("[worker] Could not query WebGPU adapter info:", e);
            }
        }

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

        // Try to get GPU memory limit for the error message
        let gpuLimitInfo = "";
        try {
            if (navigator.gpu) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const maxBuf = adapter.limits.maxBufferSize;
                    gpuLimitInfo = ` (device maxBufferSize: ${(maxBuf / 1024 / 1024).toFixed(0)} MB)`;
                }
            }
        } catch (_) { /* ignore */ }

        let userMessage;
        if (msg.includes("bad_alloc") || msg.includes("out of memory") || msg.includes("OOM")) {
            userMessage = `Out of GPU memory loading ${dtype.toUpperCase()} model${gpuLimitInfo}. ` +
                (dtype !== "q4"
                    ? `Try selecting Q4 quantization (~5.4 GB) instead of ${dtype.toUpperCase()}.`
                    : `This 7B-parameter model requires more GPU memory than this device can allocate. Try a desktop computer with 16+ GB RAM.`);
        } else {
            userMessage = `WebGPU model loading failed: ${msg}. ` +
                `Try using a browser with WebGPU support (Chrome 113+).`;
        }

        self.postMessage({
            type: "error",
            data: { message: userMessage },
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

// Repetition penalty: frequency-based (scales with number of occurrences)
const REPETITION_PENALTY = 1.3; // base multiplicative penalty per occurrence
const MAX_NGRAM_REPEAT = 3; // block 4-grams that have appeared this many times

/**
 * Greedy argmax over the last position's logits, with token suppression,
 * frequency-based repetition penalty, and n-gram blocking.
 * @param {object} logits - ORT tensor with shape [1, seqLen, vocabSize]
 * @param {number[]} generatedTokens - tokens generated so far
 */
function greedySample(logits, generatedTokens = []) {
    const data = logits.data;
    const vocabSize = logits.dims[2];
    const seqLen = logits.dims[1];

    const offset = (seqLen - 1) * vocabSize;

    // Count token frequencies for frequency-based penalty
    const tokenCounts = new Map();
    for (const t of generatedTokens) {
        tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
    }

    // N-gram blocking: find tokens that would create a repeated 4-gram
    const blockedTokens = new Set();
    if (generatedTokens.length >= 3) {
        const lastTrigram = generatedTokens.slice(-3);
        // Count how many times this trigram appears in generated tokens
        for (let i = 0; i <= generatedTokens.length - 4; i++) {
            if (generatedTokens[i] === lastTrigram[0] &&
                generatedTokens[i + 1] === lastTrigram[1] &&
                generatedTokens[i + 2] === lastTrigram[2]) {
                // The token that followed this trigram before
                const nextT = generatedTokens[i + 3];
                // Count occurrences of this 4-gram
                let count = 0;
                for (let j = 0; j <= generatedTokens.length - 4; j++) {
                    if (generatedTokens[j] === lastTrigram[0] &&
                        generatedTokens[j + 1] === lastTrigram[1] &&
                        generatedTokens[j + 2] === lastTrigram[2] &&
                        generatedTokens[j + 3] === nextT) {
                        count++;
                    }
                }
                if (count >= MAX_NGRAM_REPEAT) {
                    blockedTokens.add(nextT);
                }
            }
        }
    }

    let maxVal = -Infinity;
    let maxIdx = 0;

    for (let i = 0; i < vocabSize; i++) {
        let score = data[offset + i];

        // Suppress specific tokens
        if (SUPPRESSED_TOKENS.includes(i)) {
            score -= SUPPRESSION_PENALTY;
        }

        // N-gram blocking: hard suppress tokens that create repeated 4-grams
        if (blockedTokens.has(i)) {
            score -= 100.0;
        }

        // Frequency-based repetition penalty
        const count = tokenCounts.get(i) || 0;
        if (count > 0) {
            const penalty = Math.pow(REPETITION_PENALTY, count);
            score = score > 0 ? score / penalty : score * penalty;
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

        // Debug: log top-5 logits for each step
        {
            const logits = result.logits;
            const data = logits.data;
            const vocabSize = logits.dims[2];
            const seqLen = logits.dims[1];
            const offset = (seqLen - 1) * vocabSize;

            if (step === 0) {
                console.log(`[worker] Logits shape: ${JSON.stringify(logits.dims)}, dtype: ${logits.type}`);
            }

            // Top-5 tokens
            const scores = [];
            for (let i = 0; i < vocabSize; i++) {
                scores.push({ token: i, score: data[offset + i] });
            }
            scores.sort((a, b) => b.score - a.score);
            const top5 = scores.slice(0, 5).map(s => {
                const decoded = tokenizer.decode([s.token], { skip_special_tokens: false });
                return `${s.token}("${decoded}")=${s.score.toFixed(2)}`;
            });
            console.log(`[worker] Step ${step + 1} top-5: ${top5.join(", ")}`);

            // Check for NaN/Inf on first step
            if (step === 0) {
                let nanCount = 0, infCount = 0;
                for (let i = 0; i < vocabSize; i++) {
                    if (isNaN(data[offset + i])) nanCount++;
                    if (!isFinite(data[offset + i])) infCount++;
                }
                if (nanCount > 0 || infCount > 0) {
                    console.error(`[worker] LOGIT CORRUPTION: NaN=${nanCount}, Inf=${infCount} out of ${vocabSize}`);
                }
            }
        }

        const nextToken = greedySample(result.logits, generatedTokens);

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
    console.log("[worker] hasKVCacheInputs:", hasKVCacheInputs());
    console.log("[worker] Decoder inputs:", JSON.stringify(decoderSession.inputNames));
    console.log("[worker] Decoder outputs:", JSON.stringify(decoderSession.outputNames));
    let result = await runDecoder(speechEmbeddings, promptTokenIds, null);

    // Log prefill output keys and shapes for debugging
    for (const [name, tensor] of Object.entries(result)) {
        console.log(`[worker] Prefill output: ${name} shape=${JSON.stringify(tensor.dims)} dtype=${tensor.type}`);
    }

    let nextToken = greedySample(result.logits, generatedTokens);
    let pastKeyValues = extractPastKeyValues(result);
    console.log(`[worker] KV-cache entries: ${Object.keys(pastKeyValues).length}`);

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

        // Log first decode step details for debugging
        if (step === 1) {
            for (const [name, tensor] of Object.entries(result)) {
                console.log(`[worker] Decode step 2 output: ${name} shape=${JSON.stringify(tensor.dims)} dtype=${tensor.type}`);
            }
        }

        nextToken = greedySample(result.logits, generatedTokens);
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
        console.log("[worker] Audio length:", audioData.length, "samples,", (audioData.length / 24000).toFixed(2), "seconds");
        const speechEmbeddings = await encodeSpeech(audioData);
        console.log("[worker] Speech embeddings shape:", speechEmbeddings.dims, "dtype:", speechEmbeddings.type);

        // Debug: check speech embedding statistics
        const embData = speechEmbeddings.data;
        let min = Infinity, max = -Infinity, sum = 0, nanCount = 0, zeroCount = 0;
        for (let i = 0; i < embData.length; i++) {
            const v = embData[i];
            if (isNaN(v)) { nanCount++; continue; }
            if (v === 0) zeroCount++;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        const mean = sum / embData.length;
        console.log(`[worker] Speech embeddings stats: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, NaN=${nanCount}, zeros=${zeroCount}/${embData.length}`);

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
