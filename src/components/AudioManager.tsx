import React, { useState, useRef, useCallback } from "react";
import { SAMPLE_RATE } from "../utils/Constants";

interface AudioManagerProps {
    onAudioReady: (audioData: Float32Array) => void;
    isTranscribing: boolean;
}

export default function AudioManager({
    onAudioReady,
    isTranscribing,
}: AudioManagerProps) {
    const [isRecording, setIsRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const processAudioBlob = useCallback(
        async (blob: Blob, name?: string) => {
            const url = URL.createObjectURL(blob);
            setAudioUrl(url);
            if (name) setFileName(name);

            const arrayBuffer = await blob.arrayBuffer();
            const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

            // Get mono channel
            const channelData = audioBuffer.getChannelData(0);

            // Resample to target sample rate if needed
            let audioData: Float32Array;
            if (audioBuffer.sampleRate !== SAMPLE_RATE) {
                const offlineCtx = new OfflineAudioContext(
                    1,
                    Math.ceil(
                        (channelData.length * SAMPLE_RATE) /
                            audioBuffer.sampleRate,
                    ),
                    SAMPLE_RATE,
                );
                const source = offlineCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineCtx.destination);
                source.start();
                const resampled = await offlineCtx.startRendering();
                audioData = resampled.getChannelData(0);
            } else {
                audioData = new Float32Array(channelData);
            }

            onAudioReady(audioData);
            audioCtx.close();
        },
        [onAudioReady],
    );

    const handleFileUpload = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setFileName(file.name);
            processAudioBlob(file, file.name);
        },
        [processAudioBlob],
    );

    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, {
                    type: "audio/webm",
                });
                processAudioBlob(blob, "Recording");
                stream.getTracks().forEach((track) => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Failed to start recording:", err);
        }
    }, [processAudioBlob]);

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    }, []);

    return (
        <div className="flex flex-col items-center gap-4 w-full max-w-lg">
            <div className="flex gap-3">
                <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
                    Upload Audio
                    <input
                        type="file"
                        accept="audio/*"
                        onChange={handleFileUpload}
                        className="hidden"
                        disabled={isTranscribing}
                    />
                </label>

                <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isTranscribing}
                    className={`px-4 py-2 rounded-lg transition-colors ${
                        isRecording
                            ? "bg-red-600 hover:bg-red-700 text-white"
                            : "bg-gray-200 hover:bg-gray-300 text-gray-800"
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                    {isRecording ? "Stop Recording" : "Record"}
                </button>
            </div>

            {fileName && (
                <p className="text-sm text-gray-500">
                    {fileName}
                </p>
            )}

            {audioUrl && (
                <audio controls src={audioUrl} className="w-full" />
            )}
        </div>
    );
}
