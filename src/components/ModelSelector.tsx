import { useState, useEffect } from "react";
import {
    DecodeMode,
    DType,
    ModelConfig,
    AVAILABLE_DTYPES,
} from "../utils/Constants";

interface ModelSelectorProps {
    onLoadModel: (config: ModelConfig) => void;
}

const DECODE_MODE_OPTIONS: { value: DecodeMode; label: string; description: string }[] = [
    { value: "no-kvcache", label: "Without KV-cache", description: "Re-runs full decoder each step (simpler, slower)" },
    { value: "kvcache", label: "With KV-cache", description: "Caches key/values for faster autoregressive decoding" },
];

const DTYPE_LABELS: Record<DType, string> = {
    int8: "INT8 (~9 GB)",
    q4: "Q4 (~5.4 GB)",
    fp16: "FP16 (~15 GB)",
};

function isMobileDevice(): boolean {
    return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export default function ModelSelector({ onLoadModel }: ModelSelectorProps) {
    const [decodeMode, setDecodeMode] = useState<DecodeMode>("kvcache");
    const [dtype, setDType] = useState<DType>("q4");
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        setIsMobile(isMobileDevice());
    }, []);

    return (
        <div className="w-full max-w-md bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800">
                Model Configuration
            </h2>

            {isMobile && (
                <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                    <p className="font-medium">Mobile device detected</p>
                    <p className="mt-1 text-xs text-amber-700">
                        This 7B-parameter model requires significant GPU memory and may crash on mobile devices.
                        For best results, use a desktop computer with 16+ GB RAM.
                    </p>
                </div>
            )}

            <div className="space-y-3">
                <div>
                    <label className="block text-sm font-medium text-gray-600 mb-1">
                        Decode Mode
                    </label>
                    <select
                        value={decodeMode}
                        onChange={(e) =>
                            setDecodeMode(e.target.value as DecodeMode)
                        }
                        className="w-full rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    >
                        {DECODE_MODE_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                                {opt.label}
                            </option>
                        ))}
                    </select>
                    <p className="mt-1 text-xs text-gray-400">
                        {DECODE_MODE_OPTIONS.find((o) => o.value === decodeMode)?.description}
                    </p>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-600 mb-1">
                        Quantization
                    </label>
                    <select
                        value={dtype}
                        onChange={(e) => setDType(e.target.value as DType)}
                        className="w-full rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    >
                        {AVAILABLE_DTYPES.map((d) => (
                            <option key={d} value={d}>
                                {DTYPE_LABELS[d]}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            <button
                onClick={() => onLoadModel({ decodeMode, dtype })}
                className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
            >
                Load Model
            </button>
        </div>
    );
}
