package com.ml.breans;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * BreansReg - Linear & Logistic Regression ML
 * Author: Zine El Abidine Falouti
 * License: Open Source
 */

public class BreansReg {

    // Data split container
    public static class DataSplit {
        public double[][] X;
        public double[][] Y;

        public DataSplit(double[][] X, double[][] Y) {
            this.X = X;
            this.Y = Y;
        }
    }

    // CSV Loader to Prep the DataSet before Training
    public static class CSVLoader {
        public double[][] read(String csvName) {
            ArrayList<double[]> matrix = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(csvName))) {
                String line;
                boolean firstLine = true;
                while ((line = br.readLine()) != null) {
                    if (firstLine) {
                        firstLine = false;
                        continue;
                    }
                    String[] parts = line.split(",");
                    double[] row = new double[parts.length];
                    for (int i = 0; i < parts.length; i++) {
                        row[i] = Double.parseDouble(parts[i]);
                    }
                    matrix.add(row);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return matrix.toArray(new double[0][]);
        }

        public DataSplit split(double[][] matrix, int[] outputCols) {
            int rows = matrix.length;
            int totalCols = matrix[0].length;
            int outCount = outputCols.length;
            int inCount = totalCols - outCount;

            double[][] X = new double[rows][inCount];
            double[][] Y = new double[rows][outCount];

            for (int i = 0; i < rows; i++) {
                int xi = 0, yi = 0;
                for (int j = 0; j < totalCols; j++) {
                    boolean isOutput = false;
                    for (int outCol : outputCols) {
                        if (j == outCol) {
                            isOutput = true;
                            break;
                        }
                    }
                    if (isOutput) {
                        Y[i][yi++] = matrix[i][j];
                    } else {
                        X[i][xi++] = matrix[i][j];
                    }
                }
            }
            return new DataSplit(X, Y);
        }
    }

    // Matrix operations simplified
    public static class MatrixOps {
        public static double[][] transpose(double[][] m) {
            int r = m.length, c = m[0].length;
            double[][] t = new double[c][r];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    t[j][i] = m[i][j];
            return t;
        }

        public static double[][] multiply(double[][] A, double[][] B) {
            int rA = A.length, cA = A[0].length;
            int rB = B.length, cB = B[0].length;
            if (cA != rB)
                throw new IllegalArgumentException("Incompatible matrix sizes");
            double[][] R = new double[rA][cB];
            for (int i = 0; i < rA; i++)
                for (int j = 0; j < cB; j++)
                    for (int k = 0; k < cA; k++)
                        R[i][j] += A[i][k] * B[k][j];
            return R;
        }

        private static double[][] identity(int n) {
            double[][] I = new double[n][n];
            for (int i = 0; i < n; i++) I[i][i] = 1.0;
            return I;
        }

        public static double[][] invert(double[][] m) {
            int n = m.length;
            double[][] A = new double[n][n];
            double[][] I = identity(n);
            for (int i = 0; i < n; i++)
                System.arraycopy(m[i], 0, A[i], 0, n);

            for (int i = 0; i < n; i++) {
                double pivot = A[i][i];
                if (Math.abs(pivot) < 1e-10)
                    throw new ArithmeticException("Singular matrix");
                for (int j = 0; j < n; j++) {
                    A[i][j] /= pivot;
                    I[i][j] /= pivot;
                }
                for (int k = 0; k < n; k++) {
                    if (k == i) continue;
                    double f = A[k][i];
                    for (int j = 0; j < n; j++) {
                        A[k][j] -= f * A[i][j];
                        I[k][j] -= f * I[i][j];
                    }
                }
            }
            return I;
        }
    }

    public static class LinReg {
        private double[][] weights;  // stored weights after training

        private static double[][] addBias(double[][] X) {
            int r = X.length, c = X[0].length;
            double[][] Xb = new double[r][c + 1];
            for (int i = 0; i < r; i++) {
                Xb[i][0] = 1.0;
                System.arraycopy(X[i], 0, Xb[i], 1, c);
            }
            return Xb;
        }

        // Normal Equation train
        public void train(double[][] X, double[][] Y) {
            double[][] Xb = addBias(X);
            double[][] Xt = MatrixOps.transpose(Xb);
            double[][] XtX = MatrixOps.multiply(Xt, Xb);
            double[][] XtXinv = MatrixOps.invert(XtX);
            double[][] XtY = MatrixOps.multiply(Xt, Y);
            weights = MatrixOps.multiply(XtXinv, XtY);
        }

        // Gradient Descent train (added learningRate and epochs)
        public void train(double[][] X, double[][] Y, double learningRate, int epochs) {
            int samples = X.length;
            int features = X[0].length;
            int outputs = Y[0].length;
            double[][] Xb = addBias(X);
            weights = new double[features + 1][outputs]; // init weights to zero

            for (int epoch = 0; epoch < epochs; epoch++) {
                double[][] predictions = predictInternal(Xb); // (samples x outputs)

                // Compute error = predictions - Y
                double[][] errors = new double[samples][outputs];
                for (int i = 0; i < samples; i++) {
                    for (int o = 0; o < outputs; o++) {
                        errors[i][o] = predictions[i][o] - Y[i][o];
                    }
                }

                // Compute gradient = (1/m) * Xb^T * errors
                double[][] Xt = MatrixOps.transpose(Xb);
                double[][] grad = MatrixOps.multiply(Xt, errors);
                for (int i = 0; i < grad.length; i++) {
                    for (int j = 0; j < grad[0].length; j++) {
                        grad[i][j] /= samples;
                    }
                }

                // Update weights = weights - learningRate * grad
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] -= learningRate * grad[i][j];
                    }
                }
            }
        }

        // Internal prediction using weights and biased input
        private double[][] predictInternal(double[][] Xb) {
            int samples = Xb.length;
            int features = Xb[0].length;
            int outputs = weights[0].length;
            double[][] results = new double[samples][outputs];

            for (int i = 0; i < samples; i++) {
                for (int o = 0; o < outputs; o++) {
                    double val = 0;
                    for (int j = 0; j < features; j++) {
                        val += Xb[i][j] * weights[j][o];
                    }
                    results[i][o] = val;
                }
            }
            return results;
        }

        // Public prediction (samples without bias column)
        public double[][] predict(double[][] samples) {
            if (weights == null)
                throw new IllegalStateException("Model not trained yet!");
            double[][] samplesWithBias = addBias(samples);
            return predictInternal(samplesWithBias);
        }
    }

    public static class LogReg {
        private double[][] weights;

        private static double[][] addBias(double[][] X) {
            int r = X.length, c = X[0].length;
            double[][] Xb = new double[r][c + 1];
            for (int i = 0; i < r; i++) {
                Xb[i][0] = 1.0;
                System.arraycopy(X[i], 0, Xb[i], 1, c);
            }
            return Xb;
        }

        private static double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        // Vectorized sigmoid on matrix
        private static double[][] sigmoid(double[][] Z) {
            int r = Z.length;
            int c = Z[0].length;
            double[][] S = new double[r][c];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    S[i][j] = sigmoid(Z[i][j]);
                }
            }
            return S;
        }

        // Train logistic regression with gradient descent
        public void train(double[][] X, double[][] Y, double learningRate, int epochs) {
            int samples = X.length;
            int features = X[0].length;
            int outputs = Y[0].length;
            double[][] Xb = addBias(X);
            weights = new double[features + 1][outputs]; // initialized to zero

            for (int epoch = 0; epoch < epochs; epoch++) {
                // Compute linear combination: Z = Xb * weights
                double[][] Z = new double[samples][outputs];
                for (int i = 0; i < samples; i++) {
                    for (int o = 0; o < outputs; o++) {
                        double val = 0;
                        for (int j = 0; j < features + 1; j++) {
                            val += Xb[i][j] * weights[j][o];
                        }
                        Z[i][o] = val;
                    }
                }

                // Apply sigmoid
                double[][] predictions = sigmoid(Z);

                // Error = predictions - Y
                double[][] errors = new double[samples][outputs];
                for (int i = 0; i < samples; i++) {
                    for (int o = 0; o < outputs; o++) {
                        errors[i][o] = predictions[i][o] - Y[i][o];
                    }
                }

                // Gradient = (1/m) * Xb^T * errors
                double[][] Xt = MatrixOps.transpose(Xb);
                double[][] grad = MatrixOps.multiply(Xt, errors);
                for (int i = 0; i < grad.length; i++) {
                    for (int j = 0; j < grad[0].length; j++) {
                        grad[i][j] /= samples;
                    }
                }

                // Update weights
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[0].length; j++) {
                        weights[i][j] -= learningRate * grad[i][j];
                    }
                }
            }
        }

        // Predict probabilities (sigmoid output)
        public double[][] predictProba(double[][] samples) {
            if (weights == null)
                throw new IllegalStateException("Model not trained yet!");
            double[][] Xb = addBias(samples);
            int samplesCount = Xb.length;
            int outputs = weights[0].length;
            double[][] Z = new double[samplesCount][outputs];
            for (int i = 0; i < samplesCount; i++) {
                for (int o = 0; o < outputs; o++) {
                    double val = 0;
                    for (int j = 0; j < weights.length; j++) {
                        val += Xb[i][j] * weights[j][o];
                    }
                    Z[i][o] = val;
                }
            }
            return sigmoid(Z);
        }

        // Predict class labels 0/1 based on 0.5 threshold
        public int[][] predict(double[][] samples) {
            double[][] proba = predictProba(samples);
            int[][] classes = new int[proba.length][proba[0].length];
            for (int i = 0; i < proba.length; i++) {
                for (int j = 0; j < proba[0].length; j++) {
                    classes[i][j] = proba[i][j] >= 0.5 ? 1 : 0;
                }
            }
            return classes;
        }
    }

    public static void main(String[] args) {
        CSVLoader loader = new CSVLoader();
        double[][] data = loader.read("regdataset.csv");
        int[] outputCols = {2, 4};  // example target columns

        DataSplit split = loader.split(data, outputCols);

        LinReg model = new LinReg();

        // Train using normal equation
        model.train(split.X, split.Y);

        // Or to train using gradient descent:
        //model.train(split.X, split.Y, 0.0001, 1000);

        // Example input for prediction: must match input columns count
        double[][] exampleToPredict = {
            {37.454011884736246, 14.203164615428776, 70.89957821641812}
        };

        double[][] prediction = model.predict(exampleToPredict);

        System.out.println("Prediction result:");
        for (int i = 0; i < prediction.length; i++) {
            for (int j = 0; j < prediction[0].length; j++) {
                System.out.printf("%.6f ", prediction[i][j]);
            }
            System.out.println();
        }
    }
}
