import React from "react";
import { Zap } from "lucide-react";

export function NetworkArchitecture({ layers, activation }) {
  const layerNames = [
    "Input (2)",
    ...layers.slice(1, -1).map((n, i) => `Hidden ${i + 1} (${n})`),
    "Output (1)",
  ];

  return (
    <div className="bg-slate-900 rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between gap-2">
        {layers.map((neurons, layerIdx) => (
          <React.Fragment key={layerIdx}>
            <div className="flex flex-col items-center flex-1">
              <div className="text-xs text-gray-400 mb-2">
                {layerNames[layerIdx]}
              </div>
              <div className="flex flex-col gap-1">
                {Array(Math.min(neurons, 6))
                  .fill(0)
                  .map((_, neuronIdx) => (
                    <div
                      key={neuronIdx}
                      className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs ${
                        layerIdx === 0
                          ? "bg-blue-500 border-blue-400"
                          : layerIdx === layers.length - 1
                          ? "bg-orange-500 border-orange-400"
                          : "bg-purple-500 border-purple-400"
                      }`}
                    >
                      {neuronIdx === 5 && neurons > 6 ? "..." : ""}
                    </div>
                  ))}
              </div>
            </div>
            {layerIdx < layers.length - 1 && (
              <div className="flex flex-col items-center px-1">
                <Zap size={14} className="text-yellow-400" />
                <span className="text-xs text-gray-500 mt-1 rotate-90 whitespace-nowrap">
                  {layerIdx === layers.length - 2 ? "sigmoid" : activation}
                </span>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
