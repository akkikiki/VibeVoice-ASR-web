import { useCallback } from "react";
import AudioManager from "./components/AudioManager";
import Transcript from "./components/Transcript";
import Progress from "./components/Progress";
import ModelSelector from "./components/ModelSelector";
import { useTranscriber } from "./hooks/useTranscriber";

function App() {
    const { status, result, transcribe, loadModel } = useTranscriber();
    const isIdle = status.status === "idle";
    const isReady = status.status === "ready";
    const isTranscribing = status.status === "transcribing";

    const handleAudioReady = useCallback(
        (audioData: Float32Array) => {
            if (!isReady) return;
            transcribe(audioData);
        },
        [isReady, transcribe],
    );

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
