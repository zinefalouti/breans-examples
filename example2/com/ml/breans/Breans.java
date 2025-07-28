package com.ml.breans;

import java.io.*;
import java.util.*;

/**
 * Breans - A light-weight Classification Neural Network
 * Author: Zine El Abidine Falouti
 * License: Open Source
 */

public class Breans {

    private static final Random rand = new Random();

    // --- Activation Functions ---
    static double relu(double x) { return Math.max(0, x); }
    static double reluDeriv(double x) { return x > 0 ? 1 : 0; }
    static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    static double sigmoidDeriv(double out) { return out * (1 - out); }
    static double tanh(double x) { return Math.tanh(x); }
    static double tanhDeriv(double out) { return 1 - out * out; }

    // Softmax returns a probability distribution
    static double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(0);
        double sum = 0;
        double[] exps = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exps[i] = Math.exp(x[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < x.length; i++) {
            exps[i] /= sum;
        }
        return exps;
    }

    // --- Layer Class ---
    public static class Layer {
        int inputSize, neuronCount;
        double[][] weights;
        double[] biases;
        double[] outputs;
        double[] inputs;
        double[] deltas;
        String activationName;

        double[][] mWeights, vWeights;
        double[] mBiases, vBiases;

        Layer(int inputSize, int neuronCount, String activationName) {
            this.inputSize = inputSize;
            this.neuronCount = neuronCount;
            this.activationName = activationName;

            weights = new double[neuronCount][inputSize];
            biases = new double[neuronCount];
            outputs = new double[neuronCount];
            deltas = new double[neuronCount];

            mWeights = new double[neuronCount][inputSize];
            vWeights = new double[neuronCount][inputSize];
            mBiases = new double[neuronCount];
            vBiases = new double[neuronCount];

            // He initialization for relu, Xavier for others
            double stddev = activationName.equalsIgnoreCase("relu") ?
                    Math.sqrt(2.0 / inputSize) : Math.sqrt(1.0 / inputSize);
            for (int i = 0; i < neuronCount; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = rand.nextGaussian() * stddev;
                }
                biases[i] = 0;  // initialize biases to zero
            }
        }

        // Forward pass for this layer
        double[] forward(double[] in) {
            inputs = in;
            double[] sums = new double[neuronCount];
            for (int i = 0; i < neuronCount; i++) {
                double sum = biases[i];
                for (int j = 0; j < inputSize; j++) {
                    sum += in[j] * weights[i][j];
                }
                sums[i] = sum;
            }
            if (activationName.equalsIgnoreCase("softmax")) {
                outputs = softmax(sums);
            } else {
                for (int i = 0; i < neuronCount; i++) {
                    outputs[i] = activate(sums[i]);
                }
            }
            return outputs;
        }

        double activate(double x) {
            switch (activationName.toLowerCase()) {
                case "sigmoid": return sigmoid(x);
                case "tanh": return tanh(x);
                case "relu": return relu(x);
                case "softmax": return x;  // handled separately
                default: return x;
            }
        }

        double activationDeriv(double out) {
            switch (activationName.toLowerCase()) {
                case "sigmoid": return sigmoidDeriv(out);
                case "tanh": return tanhDeriv(out);
                case "relu": return reluDeriv(out);
                case "softmax": return 1;  // Not used explicitly in backprop for softmax+cross-entropy
                default: return 1;
            }
        }

        void save(PrintWriter pw) {
            pw.println(neuronCount + " " + inputSize);
            pw.println(activationName);
            for (int i = 0; i < neuronCount; i++) {
                for (int j = 0; j < inputSize; j++) {
                    pw.print(weights[i][j] + " ");
                }
                pw.println();
            }
            for (int i = 0; i < neuronCount; i++) {
                pw.print(biases[i] + " ");
            }
            pw.println();
        }

        static Layer load(Scanner sc) {
            int neurons = sc.nextInt();
            int inputs = sc.nextInt();
            sc.nextLine();
            String actName = sc.nextLine().trim();

            double[][] w = new double[neurons][inputs];
            for (int i = 0; i < neurons; i++) {
                String line = sc.nextLine().trim();
                String[] parts = line.split("\\s+");
                if (parts.length != inputs) {
                    throw new RuntimeException("Wrong number of weights in row " + i);
                }
                for (int j = 0; j < inputs; j++) {
                    w[i][j] = Double.parseDouble(parts[j]);
                }
            }

            double[] b = new double[neurons];
            String biasLine = sc.nextLine().trim();
            String[] biasParts = biasLine.split("\\s+");
            if (biasParts.length != neurons) {
                throw new RuntimeException("Wrong number of biases");
            }
            for (int i = 0; i < neurons; i++) {
                b[i] = Double.parseDouble(biasParts[i]);
            }

            Layer l = new Layer(inputs, neurons, actName);
            l.weights = w;
            l.biases = b;
            return l;
        }
    }

    // Dataset container
    public static class Dataset {
        public double[][] inputs;
        public double[][] targets;
        Dataset(double[][] inputs, double[][] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }
    }

    // Layer specification for easy network creation
    public static class LayerSpec {
        int neurons;
        String activation;
        public LayerSpec(int neurons, String activation) {
            this.neurons = neurons;
            this.activation = activation;
        }
    }

    // Create network from specs
    public static Layer[] createNetwork(int inputSize, LayerSpec[] specs) {
        Layer[] layers = new Layer[specs.length];
        int currentInputSize = inputSize;
        for (int i = 0; i < specs.length; i++) {
            layers[i] = new Layer(currentInputSize, specs[i].neurons, specs[i].activation);
            currentInputSize = specs[i].neurons;
        }
        return layers;
    }

    // Run forward pass through all layers
    public static double[] runForwardPass(Layer[] network, double[] input) {
        double[] activation = input;
        for (Layer layer : network) {
            activation = layer.forward(activation);
        }
        return activation;
    }

    // Backpropagation algorithm
    static void backpropagate(Layer[] network, double[] target, double learningRate) {
        Layer outputLayer = network[network.length - 1];

        // Output layer delta (softmax + cross-entropy)
        if (outputLayer.activationName.equalsIgnoreCase("softmax")) {
            for (int i = 0; i < outputLayer.neuronCount; i++) {
                outputLayer.deltas[i] = outputLayer.outputs[i] - target[i];
            }
        } else {
            for (int i = 0; i < outputLayer.neuronCount; i++) {
                double error = target[i] - outputLayer.outputs[i];
                outputLayer.deltas[i] = error * outputLayer.activationDeriv(outputLayer.outputs[i]);
            }
        }

        // Hidden layers deltas
        for (int k = network.length - 2; k >= 0; k--) {
            Layer current = network[k];
            Layer next = network[k + 1];
            for (int i = 0; i < current.neuronCount; i++) {
                double sum = 0;
                for (int j = 0; j < next.neuronCount; j++) {
                    sum += next.weights[j][i] * next.deltas[j];
                }
                current.deltas[i] = sum * current.activationDeriv(current.outputs[i]);
            }
        }

        // Update weights and biases
        for (Layer layer : network) {
            for (int i = 0; i < layer.neuronCount; i++) {
                for (int j = 0; j < layer.inputSize; j++) {
                    layer.weights[i][j] -= learningRate * layer.deltas[i] * layer.inputs[j];
                }
                layer.biases[i] -= learningRate * layer.deltas[i];
            }
        }
    }


    //ADAM Optimization
    static void adamUpdate(Layer[] network, double learningRate, double beta1, double beta2, double epsilon, int t) {
        for (Layer layer : network) {
            for (int i = 0; i < layer.neuronCount; i++) {
                for (int j = 0; j < layer.inputSize; j++) {
                    double gradW = layer.deltas[i] * layer.inputs[j];
                    layer.mWeights[i][j] = beta1 * layer.mWeights[i][j] + (1 - beta1) * gradW;
                    layer.vWeights[i][j] = beta2 * layer.vWeights[i][j] + (1 - beta2) * gradW * gradW;
                    double mHat = layer.mWeights[i][j] / (1 - Math.pow(beta1, t));
                    double vHat = layer.vWeights[i][j] / (1 - Math.pow(beta2, t));
                    layer.weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
                double gradB = layer.deltas[i];
                layer.mBiases[i] = beta1 * layer.mBiases[i] + (1 - beta1) * gradB;
                layer.vBiases[i] = beta2 * layer.vBiases[i] + (1 - beta2) * gradB * gradB;
                double mHatB = layer.mBiases[i] / (1 - Math.pow(beta1, t));
                double vHatB = layer.vBiases[i] / (1 - Math.pow(beta2, t));
                layer.biases[i] -= learningRate * mHatB / (Math.sqrt(vHatB) + epsilon);
            }
        }
    }

    public static void trainAdam(Layer[] network, Dataset dataset, int epochs, double learningRate) {
        int n = dataset.inputs.length;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
        int t = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle
            for (int i = n - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            double totalLoss = 0;
            for (int idx : indices) {
                t++;
                runForwardPass(network, dataset.inputs[idx]);
                backpropagate(network, dataset.targets[idx], 1.0); // learning rate applied in Adam
                adamUpdate(network, learningRate, beta1, beta2, epsilon, t);
                totalLoss += crossEntropyLoss(network[network.length - 1].outputs, dataset.targets[idx]);
            }
            double accuracy = computeAccuracy(network, dataset);
            System.out.printf(Locale.US, "Epoch %d - Loss: %.6f Accuracy: %.2f%n", epoch, totalLoss / n, accuracy);
        }
    }


    // Cross-entropy loss for softmax outputs
    static double crossEntropyLoss(double[] predicted, double[] target) {
        double loss = 0;
        for (int i = 0; i < predicted.length; i++) {
            loss -= target[i] * Math.log(predicted[i] + 1e-15);
        }
        return loss;
    }

    // Compute accuracy on dataset
    static double computeAccuracy(Layer[] network, Dataset dataset) {
        int correct = 0;
        for (int i = 0; i < dataset.inputs.length; i++) {
            double[] output = runForwardPass(network, dataset.inputs[i]);
            int predicted = argMax(output);
            int actual = argMax(dataset.targets[i]);
            if (predicted == actual) correct++;
        }
        return (double) correct / dataset.inputs.length;
    }

    static int argMax(double[] array) {
        int idx = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                idx = i;
            }
        }
        return idx;
    }

    // Print predictions and metrics on dataset
    public static void evaluateDataset(Layer[] network, Dataset dataset, boolean print) {
        double totalLoss = 0;
        for (int i = 0; i < dataset.inputs.length; i++) {
            double[] output = runForwardPass(network, dataset.inputs[i]);
            if (print) {
                System.out.println("Predicted: " + Arrays.toString(output) + " Target: " + Arrays.toString(dataset.targets[i]));
            }
            totalLoss += crossEntropyLoss(output, dataset.targets[i]);
        }
        double avgLoss = totalLoss / dataset.inputs.length;
        double accuracy = computeAccuracy(network, dataset);
        if (print) {
            System.out.printf(Locale.US,"Accuracy: %.2f Loss: %.4f%n", accuracy, avgLoss);
        }
    }

    // Shuffle and train network on dataset
    public static void train(Layer[] network, Dataset dataset, int epochs, double learningRate) {
        int n = dataset.inputs.length;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle data indices
            for (int i = n - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            double totalLoss = 0;
            for (int idx : indices) {
                runForwardPass(network, dataset.inputs[idx]);
                backpropagate(network, dataset.targets[idx], learningRate);
                totalLoss += crossEntropyLoss(network[network.length - 1].outputs, dataset.targets[idx]);
            }
            totalLoss /= n;
            double accuracy = computeAccuracy(network, dataset);
            System.out.printf(Locale.US, "Epoch %d - Loss: %.6f Accuracy: %.2f%n", epoch, totalLoss, accuracy);
        }
    }

    // Save network weights and biases
    public static void saveNetwork(Layer[] network, String filename, Normalization norm) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.println(network.length);
            for (Layer layer : network) {
                layer.save(pw);
            }
            norm.save(pw);  
        }
    }


    // Load network from file
    public static class LoadedModel {
        Layer[] network;
        Normalization normalization;

        LoadedModel(Layer[] network, Normalization normalization) {
            this.network = network;
            this.normalization = normalization;
        }

        public double[] predict(double[] rawInput) {
            double[] normInput = normalization.normalizeInput(rawInput);
            return runForwardPass(network, normInput);
        }
    }


    public static LoadedModel loadNetwork(String filename) throws IOException {
        try (Scanner sc = new Scanner(new FileReader(filename))) {
            int layerCount = sc.nextInt();
            sc.nextLine();
            Layer[] network = new Layer[layerCount];
            for (int i = 0; i < layerCount; i++) {
                network[i] = Layer.load(sc);
            }
            Normalization norm = Normalization.load(sc);
            return new LoadedModel(network, norm);
        }
    }



    // --- Dataset loaders ---

    // Load CSV dataset with numeric inputs and outputs
    public static Dataset loadCSVDataset(String filename, int inputCount, int outputCount) throws IOException {
        List<double[]> inputList = new ArrayList<>();
        List<double[]> targetList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                if (firstLine) {
                    firstLine = false;
                    if (!Character.isDigit(line.trim().charAt(0))) continue;
                }

                String[] parts = line.split(",");
                if (parts.length != inputCount + outputCount) {
                    throw new RuntimeException("CSV format error: expected " + (inputCount + outputCount) + " columns, got " + parts.length);
                }

                double[] inputs = new double[inputCount];
                double[] targets = new double[outputCount];

                for (int i = 0; i < inputCount; i++) {
                    inputs[i] = Double.parseDouble(parts[i]);
                }
                for (int i = 0; i < outputCount; i++) {
                    targets[i] = Double.parseDouble(parts[inputCount + i]);
                }

                inputList.add(inputs);
                targetList.add(targets);
            }
        }

        return new Dataset(inputList.toArray(new double[0][]), targetList.toArray(new double[0][]));
    }

    // Load CSV dataset with categorical inputs and output column
    public static Dataset loadCSVDatasetWithCategorical(String filename, int outputColumn) throws IOException {
        List<String[]> rawRows = new ArrayList<>();
        Map<String, Integer> outputCategoryMap = new LinkedHashMap<>();
        Map<Integer, Map<String, Integer>> categoricalMaps = new HashMap<>();

        // First pass: read rows, collect output categories and input categorical maps
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String header = br.readLine();
            if (header == null) throw new IOException("Empty CSV file!");
            int totalColumns = header.split(",", -1).length;

            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",", -1);
                if (parts.length < totalColumns) {
                    parts = Arrays.copyOf(parts, totalColumns);
                    for (int i = 0; i < totalColumns; i++) {
                        if (parts[i] == null) parts[i] = "";
                    }
                }
                rawRows.add(parts);

                // Collect output categories
                String outputRaw = parts[outputColumn].trim();
                if (!outputCategoryMap.containsKey(outputRaw)) {
                    outputCategoryMap.put(outputRaw, outputCategoryMap.size());
                }

                // Build categorical maps for inputs
                for (int i = 0; i < totalColumns; i++) {
                    if (i == outputColumn) continue;
                    String raw = parts[i].trim();
                    try {
                        Double.parseDouble(raw.replaceAll("[^0-9.\\-]", ""));
                    } catch (NumberFormatException e) {
                        categoricalMaps.putIfAbsent(i, new LinkedHashMap<>());
                        Map<String, Integer> map = categoricalMaps.get(i);
                        if (!map.containsKey(raw)) {
                            map.put(raw, map.size());
                        }
                    }
                }
            }
        }

        int totalColumns = rawRows.get(0).length;

        // Calculate input vector size with one-hot for categorical inputs
        int inputSize = 0;
        for (int i = 0; i < totalColumns; i++) {
            if (i == outputColumn) continue;
            Map<String, Integer> catMap = categoricalMaps.get(i);
            if (catMap != null) {
                inputSize += catMap.size();
            } else {
                inputSize += 1;
            }
        }

        int numCategories = outputCategoryMap.size();

        List<double[]> inputList = new ArrayList<>();
        List<double[]> targetList = new ArrayList<>();

        // Second pass: build one-hot encoded input and target vectors
        for (String[] parts : rawRows) {
            double[] inputs = new double[inputSize];
            int inputIndex = 0;

            for (int i = 0; i < totalColumns; i++) {
                if (i == outputColumn) continue;
                String raw = parts[i].trim();

                Map<String, Integer> catMap = categoricalMaps.get(i);
                if (catMap != null) {
                    int categoryCount = catMap.size();
                    int catIdx = catMap.getOrDefault(raw, 0);
                    for (int j = 0; j < categoryCount; j++) {
                        inputs[inputIndex++] = (j == catIdx) ? 1.0 : 0.0;
                    }
                } else {
                    double value;
                    try {
                        value = Double.parseDouble(raw.replaceAll("[^0-9.\\-]", ""));
                    } catch (NumberFormatException e) {
                        value = 0;
                    }
                    inputs[inputIndex++] = value;
                }
            }

            // One-hot encode output
            String outputRaw = parts[outputColumn].trim();
            int categoryIndex = outputCategoryMap.get(outputRaw);
            double[] target = new double[numCategories];
            target[categoryIndex] = 1.0;

            inputList.add(inputs);
            targetList.add(target);
        }

        return new Dataset(inputList.toArray(new double[0][]), targetList.toArray(new double[0][]));
    }


    // Normalize dataset inputs (per feature standardization)

    public static class Normalization {
        double[] means;
        double[] stds;

        Normalization(double[] means, double[] stds) {
            this.means = means;
            this.stds = stds;
        }

        void save(PrintWriter pw) {
            pw.println(means.length);
            for (double m : means) pw.print(m + " ");
            pw.println();
            for (double s : stds) pw.print(s + " ");
            pw.println();
        }

        static Normalization load(Scanner sc) {
            sc.useLocale(Locale.US);
            int len = sc.nextInt();
            sc.nextLine();
            double[] means = new double[len];
            double[] stds = new double[len];
            for (int i = 0; i < len; i++) means[i] = sc.nextDouble();
            for (int i = 0; i < len; i++) stds[i] = sc.nextDouble();
            sc.nextLine();
            return new Normalization(means, stds);
        }

        double[] normalizeInput(double[] input) {
            double[] norm = new double[input.length];
            for (int i = 0; i < input.length; i++) {
                norm[i] = (input[i] - means[i]) / stds[i];
            }
            return norm;
        }
    }


    public static Normalization normalizeDataset(Dataset dataset) {
        int inputCount = dataset.inputs[0].length;
        double[] means = new double[inputCount];
        double[] stds = new double[inputCount];

        for (int i = 0; i < inputCount; i++) {
            double sum = 0;
            for (int j = 0; j < dataset.inputs.length; j++) {
                sum += dataset.inputs[j][i];
            }
            means[i] = sum / dataset.inputs.length;
        }

        for (int i = 0; i < inputCount; i++) {
            double sumSq = 0;
            for (int j = 0; j < dataset.inputs.length; j++) {
                double diff = dataset.inputs[j][i] - means[i];
                sumSq += diff * diff;
            }
            stds[i] = Math.sqrt(sumSq / dataset.inputs.length);
            if (stds[i] < 1e-10) stds[i] = 1;
        }

        for (int j = 0; j < dataset.inputs.length; j++) {
            for (int i = 0; i < inputCount; i++) {
                dataset.inputs[j][i] = (dataset.inputs[j][i] - means[i]) / stds[i];
            }
        }

        return new Normalization(means, stds);
    }





    // --- Main Method ---
    public static void main(String[] args) throws IOException {
           // Run Locally
    }
}
