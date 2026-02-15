import { useState } from "react";
import {
    DecodeMode,
    DType,
    ModelConfig,
    AVAILABLE_DTYPES,
    DEFAULT_MAX_NEW_TOKENS,
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

export default function ModelSelector({ onLoadModel }: ModelSelectorProps) {
    const [decodeMode, setDecodeMode] = useState<DecodeMode>("kvcache");
    const [dtype, setDType] = useState<DType>("int8");
    const [maxTokens, setMaxTokens] = useState<number>(DEFAULT_MAX_NEW_TOKENS);

    return (
        <div className="w-full max-w-md bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800">
                Model Configuration
            </h2>

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
                        className="w-full rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
                    />
                    <p className="mt-1 text-xs text-gray-400">
                        Maximum number of tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})
                    </p>
                </div>
            </div>

            <button
                onClick={() => onLoadModel({ decodeMode, dtype, maxTokens })}
                className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
            >
                Load Model
            </button>
        </div>
    );
}
