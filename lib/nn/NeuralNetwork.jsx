export class NeuralNetwork {
  constructor(inputSize, hiddenSizes, outputSize, activation, learningRate) {
    this.layers = [inputSize, ...hiddenSizes, outputSize];
    this.activation = activation;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    this.initializeWeights();
  }

  initializeWeights() {
    for (let i = 0; i < this.layers.length - 1; i++) {
      const scale = Math.sqrt(2.0 / this.layers[i]);
      const w = Array(this.layers[i])
        .fill(0)
        .map(() =>
          Array(this.layers[i + 1])
            .fill(0)
            .map(() => (Math.random() - 0.5) * 2 * scale)
        );
      const b = Array(this.layers[i + 1])
        .fill(0)
        .map(() => 0);
      this.weights.push(w);
      this.biases.push(b);
    }
  }

  activate(x, derivative = false) {
    switch (this.activation) {
      case "tanh":
        if (derivative) {
          const t = Math.tanh(x);
          return 1 - t * t;
        }
        return Math.tanh(x);

      case "relu":
        if (derivative) return x > 0 ? 1 : 0;
        return Math.max(0, x);

      case "sigmoid":
        const clipped = Math.max(-20, Math.min(20, x));
        const sig = 1 / (1 + Math.exp(-clipped));
        if (derivative) return sig * (1 - sig);
        return sig;

      case "gelu":
        if (derivative) {
          const sqrt2OverPi = 0.7978845608;
          const coeff = 0.044715;
          const x3 = x * x * x;
          const tanh_arg = sqrt2OverPi * (x + coeff * x3);
          const tanh_out = Math.tanh(tanh_arg);
          const sech2 = 1 - tanh_out * tanh_out;

          return (
            0.5 * (1 + tanh_out) +
            0.5 * x * sech2 * sqrt2OverPi * (1 + 3 * coeff * x * x)
          );
        }
        const sqrt2OverPi = 0.7978845608;
        return (
          0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)))
        );

      case "linear":
        return derivative ? 1 : x;

      default:
        return x;
    }
  }

  forward(input) {
    let activation = input;
    const activations = [input];
    const zs = [];

    for (let i = 0; i < this.weights.length; i++) {
      const z = this.matrixMultiply([activation], this.weights[i])[0].map(
        (val, idx) => val + this.biases[i][idx]
      );
      zs.push(z);

      if (i === this.weights.length - 1) {
        activation = z.map((val) => {
          const clipped = Math.max(-20, Math.min(20, val));
          return 1 / (1 + Math.exp(-clipped));
        });
      } else {
        activation = z.map((val) => this.activate(val));
      }
      activations.push(activation);
    }

    return { activations, zs };
  }

  backward(x, y, activations, zs) {
    const deltas = [];
    let delta = activations[activations.length - 1].map((a, i) => a - y[i]);
    deltas.unshift(delta);

    for (let i = this.weights.length - 2; i >= 0; i--) {
      const newDelta = Array(this.layers[i + 1]).fill(0);
      for (let j = 0; j < this.layers[i + 1]; j++) {
        let sum = 0;
        for (let k = 0; k < this.layers[i + 2]; k++) {
          sum += delta[k] * this.weights[i + 1][j][k];
        }
        newDelta[j] = sum * this.activate(zs[i][j], true);
      }
      delta = newDelta;
      deltas.unshift(delta);
    }

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

  train(x, y) {
    const { activations, zs } = this.forward(x);
    this.backward(x, y, activations, zs);

    const output = activations[activations.length - 1];
    const loss =
      output.reduce((sum, val, i) => sum + Math.pow(val - y[i], 2), 0) / 2;
    return loss;
  }

  predict(input) {
    const { activations } = this.forward(input);
    return activations[activations.length - 1][0];
  }

  matrixMultiply(a, b) {
    const result = Array(a.length)
      .fill(0)
      .map(() => Array(b[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        for (let k = 0; k < b.length; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  }
}
