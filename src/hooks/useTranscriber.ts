import { useState, useRef, useCallback, useEffect } from "react";
import { SAMPLE_RATE } from "../utils/Constants";

export interface TranscriberProgress {
    status: "loading" | "downloading" | "ready" | "transcribing" | "error";
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
        status: "loading",
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

        // Initialize model loading
        worker.postMessage({ type: "load" });

        return () => {
            worker.terminate();
        };
    }, []);

    const transcribe = useCallback((audioData: Float32Array) => {
        if (!workerRef.current) return;

        workerRef.current.postMessage({
            type: "transcribe",
            audio: audioData,
            sampleRate: SAMPLE_RATE,
        });
    }, []);

    return {
        status,
        result,
        transcribe,
    };
}
