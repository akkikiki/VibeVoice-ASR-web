import { useCallback, useEffect, useState } from "react";
import AudioManager from "./components/AudioManager";
import Transcript from "./components/Transcript";
import Progress from "./components/Progress";
import ModelSelector from "./components/ModelSelector";
import { useTranscriber } from "./hooks/useTranscriber";
import { DEFAULT_PROMPT_TEMPLATE, DEFAULT_MAX_NEW_TOKENS } from "./utils/Constants";

function App() {
    const { status, result, transcribe, loadModel, stopGeneration } = useTranscriber();
    const isIdle = status.status === "idle";
    const isReady = status.status === "ready";
    const isTranscribing = status.status === "transcribing";
    const [promptTemplate, setPromptTemplate] = useState(DEFAULT_PROMPT_TEMPLATE);
    const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_NEW_TOKENS);
    const [showSettings, setShowSettings] = useState(false);
    const [webgpuSupported, setWebgpuSupported] = useState<boolean | null>(null);

    useEffect(() => {
        async function checkWebGPU() {
            try {
                if (!navigator.gpu) {
                    setWebgpuSupported(false);
                    return;
                }
                const adapter = await navigator.gpu.requestAdapter();
                setWebgpuSupported(!!adapter);
            } catch {
                setWebgpuSupported(false);
            }
        }
        checkWebGPU();
    }, []);

    const handleAudioReady = useCallback(
        (audioData: Float32Array) => {
            if (!isReady) return;
            transcribe(audioData, promptTemplate, maxTokens);
        },
        [isReady, transcribe, promptTemplate],
    );

    if (webgpuSupported === false) {
        return (
            <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4 py-8">
                <div className="w-full max-w-md bg-white rounded-xl shadow-sm border border-red-200 p-6 text-center space-y-3">
                    <h1 className="text-xl font-bold text-gray-900">
                        VibeVoice ASR
                    </h1>
                    <p className="text-red-600 font-medium">
                        WebGPU is not supported on this browser.
                    </p>
                    <p className="text-sm text-gray-500">
                        This app requires WebGPU to run large speech recognition models in the browser.
                        Please use a desktop browser with WebGPU support:
                    </p>
                    <ul className="text-sm text-gray-600 text-left list-disc list-inside space-y-1">
                        <li>Chrome 113+ (desktop)</li>
                        <li>Edge 113+ (desktop)</li>
                        <li>Chrome on Android (behind flag)</li>
                    </ul>
                    <p className="text-xs text-gray-400 pt-2">
                        On Android Chrome, try enabling <code className="bg-gray-100 px-1 rounded">chrome://flags/#enable-unsafe-webgpu</code>
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center px-4 py-8">
            <header className="mb-8 text-center">
                <h1 className="text-3xl font-bold text-gray-900">
                    VibeVoice ASR
                </h1>
                <p className="text-gray-500 mt-1">
                    Browser-based speech recognition powered by{" "}
                    <a
                        href="https://huggingface.co/microsoft/VibeVoice-ASR"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline"
                    >
                        VibeVoice-ASR
                    </a>{" "}
                    + Transformers.js + WebGPU
                </p>
            </header>

            <main className="flex flex-col items-center gap-6 w-full">
                {isIdle ? (
                    <ModelSelector onLoadModel={loadModel} />
                ) : (
                    <>
                        <Progress status={status} />

                        <AudioManager
                            onAudioReady={handleAudioReady}
                            isTranscribing={!isReady || isTranscribing}
                        />

                        {isTranscribing && (
                            <button
                                onClick={stopGeneration}
                                className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 transition-colors"
                            >
                                Stop Generation
                            </button>
                        )}

                        <div className="w-full max-w-2xl">
                            <button
                                onClick={() => setShowSettings(!showSettings)}
                                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors shadow-sm"
                            >
                                <span className="text-base">{showSettings ? "\u25B2" : "\u25BC"}</span>
                                {showSettings ? "Hide" : "Show"} Generation Settings
                            </button>
                            {showSettings && (
                                <div className="mt-3 space-y-4 rounded-lg border border-gray-200 bg-white p-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-600 mb-1">
                                            Max Tokens
                                        </label>
                                        <input
                                            type="number"
                                            value={maxTokens}
                                            onChange={(e) => setMaxTokens(Math.max(1, parseInt(e.target.value) || 1))}
                                            min={1}
                                            max={131072}
                                            className="w-32 rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                                        />
                                        <span className="ml-2 text-xs text-gray-400">
                                            (default: {DEFAULT_MAX_NEW_TOKENS})
                                        </span>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-gray-600 mb-1">
                                            Prompt Template
                                        </label>
                                        <textarea
                                            value={promptTemplate}
                                            onChange={(e) => setPromptTemplate(e.target.value)}
                                            rows={6}
                                            className="w-full rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-xs font-mono text-gray-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                                        />
                                        <p className="mt-1 text-xs text-gray-400">
                                            Use <code className="bg-gray-200 px-1 rounded">{"{duration}"}</code> as placeholder for audio duration in seconds.
                                        </p>
                                        {promptTemplate !== DEFAULT_PROMPT_TEMPLATE && (
                                            <button
                                                onClick={() => setPromptTemplate(DEFAULT_PROMPT_TEMPLATE)}
                                                className="mt-1 text-xs text-blue-600 hover:underline"
                                            >
                                                Reset to default
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        <Transcript
                            result={result}
                            isTranscribing={isTranscribing}
                        />
                    </>
                )}
            </main>

            <footer className="mt-auto pt-8 pb-4 text-center text-xs text-gray-400">
                Models run entirely in your browser via WebGPU / WASM.
                No audio is sent to any server.
            </footer>
        </div>
    );
}

export default App;
