package com.ml.breans;

import java.util.*;

/**
 * BreansMrk - A Markov Chain Framework
 * Author: Zine El Abidine Falouti
 * License: Open Source
 */

public class BreansMrk {

    /** Represents a single state in the Markov Chain */
    public static class Node {
        private final String name;
        private final Map<Node, Double> transitions = new LinkedHashMap<>();

        public Node(String name) {
            this.name = name;
        }

        public String getName() { return name; }

        /** Add a transition with a given probability */
        public void addTransition(Node target, double probability) {
            transitions.put(target, probability);
        }

        /** Ensure probabilities sum to 1 (optional) */
        public void normalizeTransitions() {
            double sum = transitions.values().stream().mapToDouble(Double::doubleValue).sum();
            if (sum == 0) return;
            for (Node key : transitions.keySet()) {
                transitions.put(key, transitions.get(key) / sum);
            }
        }

        /** Randomly pick the next node based on transition probabilities */
        public Node next() {
            double rand = Math.random();
            double cumulative = 0;
            for (Map.Entry<Node, Double> entry : transitions.entrySet()) {
                cumulative += entry.getValue();
                if (rand <= cumulative) return entry.getKey();
            }
            return null; // Shouldn't happen if normalized correctly
        }

        public Map<Node, Double> getTransitions() {
            return Collections.unmodifiableMap(transitions);
        }
    }

    /** Represents a Markov Chain of interconnected Nodes */
    public static class Chain {
        private final List<Node> nodes = new ArrayList<>();
        private Node current;

        public Chain(Node start) {
            current = start;
            if (start != null && !nodes.contains(start)) nodes.add(start);
        }

        /** Add a node to the chain */
        public void addNode(Node node) {
            if (!nodes.contains(node)) nodes.add(node);
        }

        /** Move one step in the Markov chain */
        public void step() {
            if (current != null) {
                current = current.next();
            }
        }

        public Node getCurrentNode() { return current; }

        /** Calculate the transition matrix (n x n) */
        public double[][] getTransitionMatrix() {
            int n = nodes.size();
            double[][] matrix = new double[n][n];
            for (int i = 0; i < n; i++) {
                Node row = nodes.get(i);
                for (int j = 0; j < n; j++) {
                    Node col = nodes.get(j);
                    matrix[i][j] = row.getTransitions().getOrDefault(col, 0.0);
                }
            }
            return matrix;
        }

        /** Print transition matrix in human-readable format */
        public void printTransitionMatrix() {
            double[][] matrix = getTransitionMatrix();
            System.out.print("      ");
            for (Node node : nodes) {
                System.out.printf("%8s", node.getName());
            }
            System.out.println();
            for (int i = 0; i < nodes.size(); i++) {
                System.out.printf("%6s", nodes.get(i).getName());
                for (int j = 0; j < nodes.size(); j++) {
                    System.out.printf("%8.3f", matrix[i][j]);
                }
                System.out.println();
            }
        }
    }

    public static void main(String[] args) {
        //For Local Testing
    }
}
