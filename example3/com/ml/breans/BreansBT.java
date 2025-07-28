package com.ml.breans;


/**
 * BreansBT - A Behavior Tree Framework
 * Author: Zine El Abidine Falouti
 * License: Open Source
 */

public class BreansBT {

    // --- Status Enum for BT node results ---
    public enum Status {
        RUNNING,
        SUCCESS,
        FAILURE
    }

    // --- Interface for all Behavior Tree nodes ---
    public interface BTNode {
        Status tick();
        void reset();
    }

    // --- Non-blocking timer node ---
    public static class FxTimer {
        private long startTime = -1;
        private final long durationMs;

        public FxTimer(double timeInSeconds) {
            this.durationMs = (long) (timeInSeconds * 1000);
        }

        public boolean isFinished() {
            long now = System.currentTimeMillis();
            if (startTime == -1) startTime = now;
            return now - startTime >= durationMs;
        }

        public void reset() {
            startTime = -1;
        }
    }

    public static class TimerNode implements BTNode {
        private FxTimer timer;

        public TimerNode(double seconds) {
            timer = new FxTimer(seconds);
        }

        @Override
        public Status tick() {
            return timer.isFinished() ? Status.SUCCESS : Status.RUNNING;
        }

        @Override
        public void reset() {
            timer.reset();
        }
    }

    // --- Action node wraps any function returning boolean ---
    public static class ActionNode implements BTNode {
        private final Runnable action;
        private boolean done = false;

        public ActionNode(Runnable action) {
            this.action = action;
        }

        @Override
        public Status tick() {
            if (!done) {
                action.run();
                done = true;
                return Status.SUCCESS;
            }
            return Status.SUCCESS;
        }

        @Override
        public void reset() {
            done = false;
        }
    }

    // --- Condition node wraps a boolean supplier ---
    public static class ConditionNode implements BTNode {
        private final BooleanSupplier condition;

        public ConditionNode(BooleanSupplier condition) {
            this.condition = condition;
        }

        @Override
        public Status tick() {
            return condition.getAsBoolean() ? Status.SUCCESS : Status.FAILURE;
        }

        @Override
        public void reset() {
            // nothing to reset
        }
    }

    // --- Composite Sequence Node ---
    public static class SequenceNode implements BTNode {
        private final BTNode[] children;
        private int current = 0;

        public SequenceNode(BTNode... nodes) {
            children = nodes;
        }

        @Override
        public Status tick() {
            while (current < children.length) {
                Status status = children[current].tick();
                if (status == Status.RUNNING) return Status.RUNNING;
                if (status == Status.FAILURE) {
                    return Status.FAILURE;
                }
                current++; // success, move to next
            }
            return Status.SUCCESS;
        }

        @Override
        public void reset() {
            current = 0;
            for (BTNode child : children) {
                child.reset();
            }
        }
    }

    // --- Composite Selector Node ---
    public static class SelectorNode implements BTNode {
        private final BTNode[] children;
        private int current = 0;

        public SelectorNode(BTNode... nodes) {
            children = nodes;
        }

        @Override
        public Status tick() {
            while (current < children.length) {
                Status status = children[current].tick();
                if (status == Status.RUNNING) return Status.RUNNING;
                if (status == Status.SUCCESS) {
                    return Status.SUCCESS;
                }
                current++; // failure, try next
            }
            return Status.FAILURE;
        }

        @Override
        public void reset() {
            current = 0;
            for (BTNode child : children) {
                child.reset();
            }
        }
    }

    // --- Your Fx class (utility functions) ---
    public static final class Fx {
        private Fx(){} // Prevent instantiation

        public static boolean Compare(double x, double y) {
            return x > y;
        }

        public static boolean CompareTol(double x, double y, double tolerance) {
            return Math.abs(x - y) <= tolerance;
        }

        public static boolean EqualTo(double x, double y) {
            return x == y;
        }

        public static boolean IsInRange(double value, double min, double max) {
            return value >= min && value <= max;
        }

        public static boolean Chance(double probability) {
            if (probability < 0 || probability > 1)
                throw new IllegalArgumentException("Probability must be between 0 and 1");
            return Math.random() < probability;
        }

        public static boolean NotEqualTo(double x, double y) {
            return x != y;
        }

        public static boolean AND(boolean... conditions) {
            for (boolean cond : conditions) {
                if (!cond) return false;
            }
            return true;
        }

        public static boolean OR(boolean... conditions) {
            for (boolean cond : conditions) {
                if (cond) return true;
            }
            return false;
        }

        public static boolean NOT(boolean condition) {
            return !condition;
        }
    }

    public static void main(String[] args) {
        //For Local Testing
    }



    // For ConditionNode functional interface support
    @FunctionalInterface
    public interface BooleanSupplier {
        boolean getAsBoolean();
    }

}

