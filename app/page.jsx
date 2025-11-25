"use client";

import React, { useState, useEffect, useRef } from "react";
import { NeuralNetwork } from "@/lib/nn/NeuralNetwork";
import { generateDataset } from "@/lib/nn/generateDataset";
import { NetworkArchitecture } from "@/components/NetworkArchitecture";
import { DecisionCanvas } from "@/components/DecisionCanvas";
import { TrainingPanel } from "@/components/TrainingPanel";

export default function NeuralNetworkPlayground() {
  const [dataset, setDataset] = useState("xor");
  const [activation, setActivation] = useState("tanh");
  const [learningRate, setLearningRate] = useState(0.03);
  const [hiddenLayers, setHiddenLayers] = useState([4]);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [maxEpochs, setMaxEpochs] = useState(1000);
  const [loss, setLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [data, setData] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  const networkRef = useRef(null);
  const animationRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    const newData = generateDataset(dataset, 200);
    setData(newData);
    resetNetwork();
  }, [dataset]);

  const resetNetwork = () => {
    setIsTraining(false);
    setEpoch(0);
    setLoss(0);
    setAccuracy(0);
    setElapsedTime(0);
    setStartTime(null);

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    networkRef.current = new NeuralNetwork(
      2,
      hiddenLayers,
      1,
      activation,
      learningRate
    );
  };

  useEffect(() => {
    resetNetwork();
  }, [activation, learningRate, hiddenLayers]);

  useEffect(() => {
    if (isTraining && !startTime) {
      const now = Date.now();
      setStartTime(now);
      timerRef.current = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - now) / 1000));
      }, 100);
    } else if (!isTraining && timerRef.current) {
      clearInterval(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isTraining]);

  const calculateAccuracy = () => {
    if (!networkRef.current || data.length === 0) return 0;

    let correct = 0;
    data.forEach((point) => {
      const prediction = networkRef.current.predict([point.x, point.y]);
      const predictedLabel = prediction > 0.5 ? 1 : 0;
      if (predictedLabel === point.label) correct++;
    });

    return (correct / data.length) * 100;
  };

  const trainStep = () => {
    if (!networkRef.current || !isTraining) return;

    let totalLoss = 0;
    data.forEach((point) => {
      const currentLoss = networkRef.current.train(
        [point.x, point.y],
        [point.label]
      );
      totalLoss += currentLoss;
    });

    const avgLoss = totalLoss / data.length;
    setLoss(avgLoss);
    setEpoch((e) => e + 1);

    if (epoch % 5 === 0) {
      const acc = calculateAccuracy();
      setAccuracy(acc);
    }

    if (epoch >= maxEpochs) {
      setIsTraining(false);
      return;
    }

    animationRef.current = requestAnimationFrame(trainStep);
  };

  useEffect(() => {
    if (isTraining) {
      trainStep();
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isTraining, epoch]);

  const addHiddenLayer = () => {
    if (hiddenLayers.length < 3) {
      setHiddenLayers([...hiddenLayers, 4]);
    }
  };

  const removeHiddenLayer = () => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.slice(0, -1));
    }
  };

  const updateLayerNeurons = (index, delta) => {
    const newLayers = [...hiddenLayers];
    newLayers[index] = Math.max(1, Math.min(8, newLayers[index] + delta));
    setHiddenLayers(newLayers);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold mb-2">
            Tinker With a <span className="text-blue-400">Neural Network</span>{" "}
            Right Here
          </h1>
          <p className="text-lg sm:text-xl text-gray-300">
            Explore how activation functions shape learning
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <TrainingPanel
            isTraining={isTraining}
            epoch={epoch}
            maxEpochs={maxEpochs}
            loss={loss}
            accuracy={accuracy}
            elapsedTime={elapsedTime}
            dataset={dataset}
            activation={activation}
            learningRate={learningRate}
            hiddenLayers={hiddenLayers}
            onToggleTraining={() => setIsTraining(!isTraining)}
            onReset={resetNetwork}
            onMaxEpochsChange={setMaxEpochs}
            onDatasetChange={setDataset}
            onActivationChange={setActivation}
            onLearningRateChange={setLearningRate}
            onAddLayer={addHiddenLayer}
            onRemoveLayer={removeHiddenLayer}
            onUpdateLayerNeurons={updateLayerNeurons}
          />

          <div className="lg:col-span-2 bg-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Network Architecture</h3>
            <NetworkArchitecture
              layers={[2, ...hiddenLayers, 1]}
              activation={activation}
            />

            <h3 className="text-lg font-semibold mb-4 mt-6">
              Decision Boundary
            </h3>
            <DecisionCanvas
              data={data}
              network={networkRef.current}
              epoch={epoch}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
