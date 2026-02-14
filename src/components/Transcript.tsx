import { TranscriberResult } from "../hooks/useTranscriber";

interface TranscriptProps {
    result: TranscriberResult | null;
    isTranscribing: boolean;
}

export default function Transcript({ result, isTranscribing }: TranscriptProps) {
    return (
        <div className="w-full max-w-lg">
            <h2 className="text-lg font-semibold mb-2 text-gray-700">
                Transcription
            </h2>
            <div className="bg-white border border-gray-200 rounded-lg p-4 min-h-[120px] shadow-sm">
                {isTranscribing && !result && (
                    <div className="flex items-center gap-2 text-gray-400">
                        <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
                        Transcribing...
                    </div>
                )}
                {result ? (
                    <div>
                        <p className="text-gray-800 whitespace-pre-wrap">
                            {result.text || "(no text)"}
                        </p>
                        {result.inferenceTime != null && (
                            <p className="text-xs text-gray-400 mt-3">
                                Inference time:{" "}
                                {(result.inferenceTime / 1000).toFixed(2)}s
                            </p>
                        )}
                    </div>
                ) : (
                    !isTranscribing && (
                        <p className="text-gray-400 italic">
                            Upload or record audio to get started.
                        </p>
                    )
                )}
            </div>
        </div>
    );
}
