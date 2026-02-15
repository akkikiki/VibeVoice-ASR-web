import { useState, useRef, useCallback, useEffect } from "react";
import { SAMPLE_RATE, ModelConfig } from "../utils/Constants";

export interface TranscriberProgress {
    status: "idle" | "loading" | "downloading" | "ready" | "transcribing" | "error";
    progress?: number;
    file?: string;
    message?: string;
    device?: "webgpu" | "wasm" | null;
}

export interface TranscriberResult {
    text: string;
    inferenceTime?: number;
}

export function useTranscriber() {
    const [status, setStatus] = useState<TranscriberProgress>({
        status: "idle",
        device: null,
    });
    const [result, setResult] = useState<TranscriberResult | null>(null);
    const workerRef = useRef<Worker | null>(null);

    useEffect(() => {
        const worker = new Worker(
            new URL("../worker.js", import.meta.url),
            { type: "module" },
        );

        worker.onmessage = (e) => {
            const { type, data } = e.data;

            switch (type) {
                case "device_info":
                    setStatus((prev) => ({
                        ...prev,
                        device: data.device,
                    }));
                    break;

                case "download_progress":
                    setStatus((prev) => ({
                        status: "downloading",
                        progress: data.progress,
                        file: data.file,
                        device: prev.device,
                    }));
                    break;

                case "loading_status":
                    setStatus((prev) => ({
                        status: "loading",
                        message: data.message,
                        device: prev.device,
                    }));
                    break;

                case "ready":
                    setStatus((prev) => ({
                        status: "ready",
                        device: prev.device,
                    }));
                    break;

                case "transcription_start":
                    setStatus((prev) => ({
                        status: "transcribing",
                        device: prev.device,
                    }));
                    setResult(null);
                    break;

                case "transcription_partial":
                    setResult({ text: data.text });
                    break;

                case "transcription_complete":
                    setStatus((prev) => ({
                        status: "ready",
                        device: prev.device,
                    }));
                    setResult({
                        text: data.text,
                        inferenceTime: data.inferenceTime,
                    });
                    break;

                case "error":
                    setStatus((prev) => ({
                        status: "error",
                        message: data.message,
                        device: prev.device,
                    }));
                    break;
            }
        };

        worker.onerror = (e) => {
            console.error("[useTranscriber] Worker error:", e);
            setStatus((prev) => ({
                status: "error",
                message: `Worker crashed: ${e.message || "Check browser console for details"}`,
                device: prev.device,
            }));
        };

        worker.onmessageerror = (e) => {
            console.error("[useTranscriber] Worker message error:", e);
            setStatus((prev) => ({
                status: "error",
                message: "Worker message deserialization failed",
                device: prev.device,
            }));
        };

        workerRef.current = worker;

        return () => {
            worker.terminate();
        };
    }, []);

    const loadModel = useCallback((config: ModelConfig) => {
        if (!workerRef.current) return;
        setStatus({ status: "loading", device: null });
        workerRef.current.postMessage({ type: "load", config });
    }, []);

    const transcribe = useCallback((audioData: Float32Array, promptTemplate?: string, maxTokens?: number) => {
        if (!workerRef.current) return;

        workerRef.current.postMessage({
            type: "transcribe",
            audio: audioData,
            sampleRate: SAMPLE_RATE,
            promptTemplate,
            maxTokens,
        });
    }, []);

    const stopGeneration = useCallback(() => {
        if (!workerRef.current) return;
        workerRef.current.postMessage({ type: "stop" });
    }, []);

    return {
        status,
        result,
        transcribe,
        loadModel,
        stopGeneration,
    };
}
