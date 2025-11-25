"use client";

import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, RotateCcw, Plus, Minus } from "lucide-react";

/**
 * A tiny neural network implementation for binary classification.
 * Input: x,y coordinates
 * Output: probability of class 1
 */
class NeuralNetwork {
  constructor(inputSize, hiddenSizes, outputSize, activation, learningRate) {
    this.layers = [inputSize, ...hiddenSizes, outputSize];
    this.activation = activation;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    this.initParams();
  }

  // random weight and bias initialization
  initParams() {
    for (let i = 0; i < this.layers.length - 1; i++) {
      const layerIn = this.layers[i];
      const layerOut = this.layers[i + 1];

      const weightMatrix = Array.from({ length: layerIn }, () =>
        Array.from({ length: layerOut }, () => (Math.random() - 0.5) * 2)
      );

      const biasVec = Array.from(
        { length: layerOut },
        () => (Math.random() - 0.5) * 0.1
      );

      this.weights.push(weightMatrix);
      this.biases.push(biasVec);
    }
  }

  // activation functions
  activate(x, derivative = false) {
    switch (this.activation) {
      case "tanh":
        if (derivative) return 1 - Math.tanh(x) ** 2;
        return Math.tanh(x);

      case "relu":
        if (derivative) return x > 0 ? 1 : 0;
        return Math.max(0, x);

      case "sigmoid": {
        const s = 1 / (1 + Math.exp(-x));
        return derivative ? s * (1 - s) : s;
      }

      case "linear":
        return derivative ? 1 : x;

      default:
        return x;
    }
  }

  forward(input) {
    let current = input;
    const activations = [input];
    const preActivations = [];

    for (let i = 0; i < this.weights.length; i++) {
      const w = this.weights[i];
      const b = this.biases[i];

      const z = this.matrixMultiply([current], w)[0].map(
        (v, idx) => v + b[idx]
      );
      preActivations.push(z);

      if (i === this.weights.length - 1) {
        current = z.map((v) => 1 / (1 + Math.exp(-v))); // sigmoid for output
      } else {
        current = z.map((v) => this.activate(v));
      }

      activations.push(current);
    }

    return { activations, preActivations };
  }

  backward(x, y, activations, preActivations) {
    const deltas = [];

    // output delta
    let delta = activations.at(-1).map((a, i) => a - y[i]);
    deltas.unshift(delta);

    // hidden layers delta
    for (let i = this.weights.length - 2; i >= 0; i--) {
      const next = Array(this.layers[i + 1]).fill(0);
      for (let j = 0; j < this.layers[i + 1]; j++) {
        let sum = 0;
        for (let k = 0; k < this.layers[i + 2]; k++) {
          sum += delta[k] * this.weights[i + 1][j][k];
        }
        next[j] = sum * this.activate(preActivations[i][j], true);
      }
      delta = next;
      deltas.unshift(delta);
    }

    // update params
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] -=
            this.learningRate * deltas[i][k] * activations[i][j];
        }
      }
      for (let j = 0; j < this.biases[i].length; j++) {
        this.biases[i][j] -= this.learningRate * deltas[i][j];
      }
    }
  }

  train(input, target) {
    const { activations, preActivations } = this.forward(input);
    this.backward(input, target, activations, preActivations);
    const out = activations.at(-1);
    return out.reduce((sum, v, i) => sum + (v - target[i]) ** 2, 0) / 2;
  }

  predict(input) {
    const { activations } = this.forward(input);
    return activations.at(-1)[0];
  }

  matrixMultiply(a, b) {
    return a.map((row) =>
      b[0].map((_, idx) => row.reduce((sum, v, j) => sum + v * b[j][idx], 0))
    );
  }
}

// generate datasets to play with
const generateDataset = (name, n) => {
  const pts = [];
  switch (name) {
    case "circle": {
      for (let i = 0; i < n; i++) {
        const r = Math.random();
        const angle = Math.random() * Math.PI * 2;
        const x = r * Math.cos(angle);
        const y = r * Math.sin(angle);
        pts.push({ x, y, label: r < 0.5 ? 0 : 1 });
      }
      break;
    }

    case "xor": {
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        pts.push({ x, y, label: x * y > 0 ? 1 : 0 });
      }
      break;
    }

    case "spiral": {
      for (let i = 0; i < n / 2; i++) {
        const r = (i / (n / 2)) * 0.9;
        const t = (i / (n / 2)) * Math.PI * 4;
        pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), label: 0 });
        pts.push({
          x: r * Math.cos(t + Math.PI),
          y: r * Math.sin(t + Math.PI),
          label: 1,
        });
      }
      break;
    }

    case "gaussian": {
      for (let i = 0; i < n / 2; i++) {
        pts.push({
          x: (Math.random() - 0.5) * 0.8 - 0.3,
          y: (Math.random() - 0.5) * 0.8,
          label: 0,
        });
        pts.push({
          x: (Math.random() - 0.5) * 0.8 + 0.3,
          y: (Math.random() - 0.5) * 0.8,
          label: 1,
        });
      }
      break;
    }
  }
  return pts;
};

export default function NeuralNetworkPlayground() {
  const [dataset, setDataset] = useState("xor");
  const [activation, setActivation] = useState("tanh");
  const [learningRate, setLearningRate] = useState(0.03);
  const [hiddenNeurons, setHiddenNeurons] = useState(4);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [data, setData] = useState([]);

  const canvasRef = useRef(null);
  const networkRef = useRef(null);
  const loopRef = useRef(null);

  useEffect(() => {
    const pts = generateDataset(dataset, 200);
    setData(pts);
    restartTraining();
  }, [dataset]);

  const restartTraining = () => {
    setIsTraining(false);
    setEpoch(0);
    setLoss(0);

    if (loopRef.current) cancelAnimationFrame(loopRef.current);

    networkRef.current = new NeuralNetwork(
      2,
      [hiddenNeurons],
      1,
      activation,
      learningRate
    );

    drawMesh();
  };

  useEffect(() => {
    restartTraining();
  }, [activation, learningRate, hiddenNeurons]);

  useEffect(() => {
    drawMesh();
  }, [data, epoch]);

  const drawMesh = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;

    ctx.clearRect(0, 0, width, height);

    // background decision surface
    if (networkRef.current) {
      const step = 5;
      for (let i = 0; i < width; i += step) {
        for (let j = 0; j < height; j += step) {
          const x = (i / width) * 2 - 1;
          const y = (j / height) * 2 - 1;
          const p = networkRef.current.predict([x, y]);
          ctx.fillStyle =
            p > 0.5
              ? `rgba(255,165,0,${p * 0.3})`
              : `rgba(65,105,225,${(1 - p) * 0.3})`;
          ctx.fillRect(i, j, step, step);
        }
      }
    }

    // training points
    for (const p of data) {
      const px = ((p.x + 1) / 2) * width;
      const py = ((p.y + 1) / 2) * height;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = p.label ? "#FF8C00" : "#4169E1";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  };

  const trainOne = () => {
    if (!networkRef.current || !isTraining) return;

    let total = 0;
    for (const p of data) {
      total += networkRef.current.train([p.x, p.y], [p.label]);
    }

    setLoss(total / data.length);
    setEpoch((e) => e + 1);

    if (epoch % 10 === 0) drawMesh();

    loopRef.current = requestAnimationFrame(trainOne);
  };

  useEffect(() => {
    if (isTraining) trainOne();
    else if (loopRef.current) cancelAnimationFrame(loopRef.current);

    return () => loopRef.current && cancelAnimationFrame(loopRef.current);
  }, [isTraining, epoch]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">
            Play With a <span className="text-blue-400">Neural Network</span> in
            Your Browser
          </h1>
          <p className="text-xl text-gray-300">
            Experiment freely — you can’t break anything here.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* sidebar */}
          <div className="lg:col-span-1 bg-slate-800 rounded-lg p-6 space-y-6">
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Training Panel</h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsTraining((prev) => !prev)}
                    className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
                  >
                    {isTraining ? <Pause size={20} /> : <Play size={20} />}
                  </button>
                  <button
                    onClick={restartTraining}
                    className="p-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition"
                  >
                    <RotateCcw size={20} />
                  </button>
                </div>
              </div>

              <div className="bg-slate-700 rounded p-3 space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Epoch:</span>
                  <span className="font-mono">
                    {epoch.toString().padStart(6, "0")}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Loss:</span>
                  <span className="font-mono">{loss.toFixed(4)}</span>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
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
                onChange={(e) => setActivation(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2"
              >
                <option value="linear">Linear (No Activation)</option>
                <option value="tanh">Tanh</option>
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Learning Rate: {learningRate}
              </label>
              <input
                type="range"
                min="0.001"
                max="0.3"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Hidden Layer Neurons
              </label>
              <div className="flex items-center gap-3">
                <button
                  onClick={() =>
                    setHiddenNeurons(Math.max(1, hiddenNeurons - 1))
                  }
                  className="p-2 bg-slate-700 hover:bg-slate-600 rounded"
                >
                  <Minus size={16} />
                </button>
                <span className="flex-1 text-center font-mono text-xl">
                  {hiddenNeurons}
                </span>
                <button
                  onClick={() =>
                    setHiddenNeurons(Math.min(8, hiddenNeurons + 1))
                  }
                  className="p-2 bg-slate-700 hover:bg-slate-600 rounded"
                >
                  <Plus size={16} />
                </button>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-700">
              <h4 className="font-semibold mb-2">
                💡 Why Activation Functions?
              </h4>
              <p className="text-sm text-gray-300">
                Try “Linear” with XOR or Spiral — the model won’t learn.
                Non-linear activations allow neural nets to learn curved
                patterns and boundaries.
              </p>
            </div>
          </div>

          {/* main visual */}
          <div className="lg:col-span-2 bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Decision Surface</h3>
            <div className="bg-slate-900 rounded-lg p-4">
              <canvas
                ref={canvasRef}
                width={600}
                height={600}
                className="w-full h-auto rounded border border-slate-700"
              />
            </div>
            <div className="mt-4 flex justify-center gap-8 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-blue-500" />
                <span>Class 0</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-orange-500" />
                <span>Class 1</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
