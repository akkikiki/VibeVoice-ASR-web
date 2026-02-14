import { TranscriberProgress } from "../hooks/useTranscriber";

interface ProgressProps {
    status: TranscriberProgress;
}

function DeviceBadge({ device }: { device?: "webgpu" | "wasm" | null }) {
    if (!device) return null;

    const isWebGPU = device === "webgpu";
    const label = isWebGPU ? "WebGPU" : "WASM";
    const className = isWebGPU
        ? "bg-green-100 text-green-700 border-green-200"
        : "bg-yellow-100 text-yellow-700 border-yellow-200";

    return (
        <span
            className={`inline-block px-2 py-0.5 text-xs font-medium rounded border ${className}`}
        >
            {label}
        </span>
    );
}

export default function Progress({ status }: ProgressProps) {
    if (status.status === "ready") return null;

    return (
        <div className="w-full max-w-lg">
            {status.status === "loading" && (
                <div className="text-center text-gray-500 space-y-1">
                    <div>{status.message || "Initializing model..."}</div>
                    <DeviceBadge device={status.device} />
                </div>
            )}

            {status.status === "downloading" && (
                <div className="space-y-2">
                    <div className="flex justify-between items-center text-sm text-gray-600">
                        <span className="flex items-center gap-2">
                            Downloading{" "}
                            {status.file ? `(${status.file})` : "model"}...
                            <DeviceBadge device={status.device} />
                        </span>
                        {status.progress != null && (
                            <span>{Math.round(status.progress)}%</span>
                        )}
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{
                                width: `${status.progress ?? 0}%`,
                            }}
                        />
                    </div>
                </div>
            )}

            {status.status === "error" && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
                    Error: {status.message || "Unknown error"}
                </div>
            )}
        </div>
    );
}
