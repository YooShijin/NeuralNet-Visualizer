import React from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Plus,
  Minus,
  Clock,
  Target,
  Zap,
  Layers,
} from "lucide-react";

export function TrainingPanel({
  isTraining,
  epoch,
  maxEpochs,
  loss,
  accuracy,
  elapsedTime,
  dataset,
  activation,
  learningRate,
  hiddenLayers,
  onToggleTraining,
  onReset,
  onMaxEpochsChange,
  onDatasetChange,
  onActivationChange,
  onLearningRateChange,
  onAddLayer,
  onRemoveLayer,
  onUpdateLayerNeurons,
}) {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="lg:col-span-1 bg-slate-800 rounded-lg p-6 space-y-6">
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Training Controls</h3>
          <div className="flex gap-2">
            <button
              onClick={onToggleTraining}
              className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition disabled:opacity-50"
              disabled={epoch >= maxEpochs}
            >
              {isTraining ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button
              onClick={onReset}
              className="p-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition"
            >
              <RotateCcw size={20} />
            </button>
          </div>
        </div>

        <div className="bg-slate-700 rounded p-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Target size={14} className="text-blue-400" />
              <span>Epoch:</span>
            </div>
            <span className="font-mono">
              {epoch} / {maxEpochs}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-yellow-400" />
              <span>Loss:</span>
            </div>
            <span className="font-mono">{loss.toFixed(4)}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Target size={14} className="text-green-400" />
              <span>Accuracy:</span>
            </div>
            <span className="font-mono">{accuracy.toFixed(1)}%</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Clock size={14} className="text-purple-400" />
              <span>Time:</span>
            </div>
            <span className="font-mono">{formatTime(elapsedTime)}</span>
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Max Epochs</label>
        <input
          type="number"
          value={maxEpochs}
          onChange={(e) =>
            onMaxEpochsChange(Math.max(1, parseInt(e.target.value) || 1000))
          }
          className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
          min="1"
          step="100"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Dataset</label>
        <select
          value={dataset}
          onChange={(e) => onDatasetChange(e.target.value)}
          className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
        >
          <option value="circle">Circle</option>
          <option value="xor">XOR</option>
          <option value="spiral">Spiral</option>
          <option value="gaussian">Gaussian</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">
          Activation Function
        </label>
        <select
          value={activation}
          onChange={(e) => onActivationChange(e.target.value)}
          className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
        >
          <option value="linear">Linear (No Activation)</option>
          <option value="tanh">Tanh</option>
          <option value="relu">ReLU</option>
          <option value="gelu">GELU</option>
          <option value="sigmoid">Sigmoid</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">
          Learning Rate: {learningRate.toFixed(3)}
        </label>
        <input
          type="range"
          min="0.001"
          max="0.3"
          step="0.001"
          value={learningRate}
          onChange={(e) => onLearningRateChange(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <Layers size={16} />
            Hidden Layers ({hiddenLayers.length})
          </label>
          <div className="flex gap-1">
            <button
              onClick={onRemoveLayer}
              disabled={hiddenLayers.length <= 1}
              className="p-1 bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50 text-xs"
            >
              Remove
            </button>
            <button
              onClick={onAddLayer}
              disabled={hiddenLayers.length >= 3}
              className="p-1 bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50 text-xs"
            >
              Add
            </button>
          </div>
        </div>

        {hiddenLayers.map((neurons, idx) => (
          <div key={idx} className="flex items-center gap-3 mb-2">
            <span className="text-xs text-gray-400 w-16">Layer {idx + 1}:</span>
            <button
              onClick={() => onUpdateLayerNeurons(idx, -1)}
              className="p-1 bg-slate-700 hover:bg-slate-600 rounded"
            >
              <Minus size={14} />
            </button>
            <span className="flex-1 text-center font-mono">{neurons}</span>
            <button
              onClick={() => onUpdateLayerNeurons(idx, 1)}
              className="p-1 bg-slate-700 hover:bg-slate-600 rounded"
            >
              <Plus size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="pt-4 border-t border-slate-700">
        <h4 className="font-semibold mb-2">Universal Approximator</h4>
        <p className="text-sm text-gray-300">
          Try Linear activation on XOR—it can't learn it! Non-linear functions
          like ReLU, GELU, and Tanh let networks capture complex patterns. GELU
          is popular in modern transformers!
        </p>
      </div>
    </div>
  );
}
