package com.ml.breans;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.io.FileWriter;
import java.awt.Desktop;
import java.io.File;

/**
 * BreansK - A Clustering ML Algorithm (Unsuprevised Machine Learning)
 * Author: Zine El Abidine Falouti
 * License: Open Source
 */

public class BreansK {

    public static class KMeansResult {
        public final int[] assignments;
        public final ArrayList<double[]> centroids;
        public final ArrayList<Double> inertiaHistory;

        public KMeansResult(int[] assignments, ArrayList<double[]> centroids, ArrayList<Double> inertiaHistory) {
            this.assignments = assignments;
            this.centroids = centroids;
            this.inertiaHistory = inertiaHistory;
        }
    }

    // Euclidean distance between two points
    public static double EDistance(double[] pointA, double[] pointB) {
        double sumDistance = 0.0;
        for (int i = 0; i < pointA.length; i++) {
            double diff = pointA[i] - pointB[i];
            sumDistance += diff * diff;
        }
        return Math.sqrt(sumDistance);
    }

    // Load CSV file into a matrix, skipping header
    public static double[][] CSVtoMatrix(String csvName) {
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

    // Initialize centroids with K-Means++ style max-min distance
    public static ArrayList<double[]> initializeCentroids(int k, double[][] data) {
        Random rand = new Random();
        ArrayList<double[]> centroids = new ArrayList<>();
        centroids.add(data[rand.nextInt(data.length)]);

        while (centroids.size() < k) {
            double maxDist = -1;
            double[] nextCentroid = null;

            for (double[] point : data) {
                double minDist = Double.MAX_VALUE;
                for (double[] centroid : centroids) {
                    double dist = EDistance(point, centroid);
                    if (dist < minDist) {
                        minDist = dist;
                    }
                }
                if (minDist > maxDist) {
                    maxDist = minDist;
                    nextCentroid = point;
                }
            }

            if (nextCentroid != null) {
                centroids.add(nextCentroid);
            } else {
                break;
            }
        }
        return centroids;
    }

    // Assign points to nearest centroid
    public static int[] assignClusters(double[][] data, ArrayList<double[]> centroids) {
        int[] assignments = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            double minDist = Double.MAX_VALUE;
            int cluster = -1;
            for (int c = 0; c < centroids.size(); c++) {
                double dist = EDistance(data[i], centroids.get(c));
                if (dist < minDist) {
                    minDist = dist;
                    cluster = c;
                }
            }
            assignments[i] = cluster;
        }
        return assignments;
    }

    // Update centroids by averaging points in clusters
    public static ArrayList<double[]> updateCentroids(double[][] data, int[] assignments, int k) {
        int dim = data[0].length;
        ArrayList<double[]> newCentroids = new ArrayList<>();
        double[][] sums = new double[k][dim];
        int[] counts = new int[k];

        for (int i = 0; i < data.length; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int j = 0; j < dim; j++) {
                sums[cluster][j] += data[i][j];
            }
        }

        for (int c = 0; c < k; c++) {
            double[] centroid = new double[dim];
            if (counts[c] == 0) {
                centroid = data[new Random().nextInt(data.length)];
            } else {
                for (int j = 0; j < dim; j++) {
                    centroid[j] = sums[c][j] / counts[c];
                }
            }
            newCentroids.add(centroid);
        }

        return newCentroids;
    }

    // Check convergence by comparing centroid shifts to tolerance
    public static boolean converged(ArrayList<double[]> oldCentroids, ArrayList<double[]> newCentroids, double tol) {
        for (int i = 0; i < oldCentroids.size(); i++) {
            if (EDistance(oldCentroids.get(i), newCentroids.get(i)) > tol) {
                return false;
            }
        }
        return true;
    }

    // Calculate inertia (sum of squared distances from points to assigned centroids)
    public static double calculateInertia(double[][] data, int[] assignments, ArrayList<double[]> centroids) {
        double total = 0.0;
        for (int i = 0; i < data.length; i++) {
            double dist = EDistance(data[i], centroids.get(assignments[i]));
            total += dist * dist;
        }
        return total;
    }

    // K-Means algorithm, returns result object with assignments, centroids, inertia history
    public static KMeansResult kMeans(double[][] data, int k, int maxIterations, double tolerance) {
        ArrayList<double[]> centroids = initializeCentroids(k, data);
        int[] assignments = new int[data.length];
        ArrayList<Double> inertiaHistory = new ArrayList<>();

        for (int iter = 0; iter < maxIterations; iter++) {
            assignments = assignClusters(data, centroids);
            ArrayList<double[]> newCentroids = updateCentroids(data, assignments, k);
            double inertia = calculateInertia(data, assignments, newCentroids);
            inertiaHistory.add(inertia);

            if (converged(centroids, newCentroids, tolerance)) {
                break;
            }
            centroids = newCentroids;
        }

        return new KMeansResult(assignments, centroids, inertiaHistory);
    }

    //Printing the Clusters
    public static void exportClustersToCSV(double[][] data, int[] assignments, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            // Write header
            for (int j = 0; j < data[0].length; j++) {
                writer.append("Feature").append(String.valueOf(j + 1)).append(",");
            }
            writer.append("Cluster\n");

            // Write data rows with cluster assignment
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    writer.append(String.valueOf(data[i][j])).append(",");
                }
                writer.append(String.valueOf(assignments[i])).append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //Generate and Export HTML
    public static String getRandomColor() {
        Random rand = new Random();
        int r = rand.nextInt(256);
        int g = rand.nextInt(256);
        int b = rand.nextInt(256);
        return String.format("#%02X%02X%02X", r, g, b);
    }

    public static String[] generateRandomColors(int k) {
        String[] colors = new String[k];
        for (int i = 0; i < k; i++) {
            colors[i] = getRandomColor();
        }
        return colors;
    }

    public static void exportClustersToHTML(double[][] data, int[] assignments, int k, String filename) {
        String[] colors = generateRandomColors(k);

        try (FileWriter writer = new FileWriter(filename)) {
            writer.append("<html><head><style>");
            writer.append("table {border-collapse: collapse;}");
            writer.append("td, th {border: 1px solid black; padding: 5px;}");
            writer.append("</style></head><body>\n");
            writer.append("<table>\n<tr>");

            // Header row
            for (int j = 0; j < data[0].length; j++) {
                writer.append("<th>").append("Feature ").append(String.valueOf(j + 1)).append("</th>");
            }
            writer.append("<th>Cluster</th></tr>\n");

            // Data rows with background color for cluster
            for (int i = 0; i < data.length; i++) {
                int cluster = assignments[i];
                String color = colors[cluster];
                writer.append("<tr style='background-color:").append(color).append("'>");
                for (int j = 0; j < data[i].length; j++) {
                    writer.append("<td>").append(String.format("%.4f", data[i][j])).append("</td>");
                }
                writer.append("<td>").append(String.valueOf(cluster)).append("</td>");
                writer.append("</tr>\n");
            }

            writer.append("</table>\n</body></html>");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Open the HTML file automatically
        try {
            File htmlFile = new File(filename);
            if (Desktop.isDesktopSupported()) {
                Desktop.getDesktop().browse(htmlFile.toURI());
            } else {
                System.out.println("Desktop is not supported. Open " + filename + " manually.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        //To Test in Local
    }
}
