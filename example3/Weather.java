/*
 * A Weather Prediction Example Using Breans Markov Chains
 */

//Import Breans Markov Chains
import com.ml.breans.BreansMrk;

public class Weather {
    public static void main(String[] args) {
        // Create weather states
        BreansMrk.Node sunny = new BreansMrk.Node("Sunny");
        BreansMrk.Node rainy = new BreansMrk.Node("Rainy");
        BreansMrk.Node cloudy = new BreansMrk.Node("Cloudy");

        // Define transitions for Sunny
        sunny.addTransition(sunny, 0.6);  // 60% chance stay sunny
        sunny.addTransition(cloudy, 0.3); // 30% chance to become cloudy
        sunny.addTransition(rainy, 0.1);  // 10% chance to rain

        // Define transitions for Rainy
        rainy.addTransition(rainy, 0.5);  // 50% chance stay rainy
        rainy.addTransition(cloudy, 0.3); // 30% chance to cloudy
        rainy.addTransition(sunny, 0.2);  // 20% chance sunny

        // Define transitions for Cloudy
        cloudy.addTransition(cloudy, 0.4); // 40% chance stay cloudy
        cloudy.addTransition(sunny, 0.4);  // 40% chance sunny
        cloudy.addTransition(rainy, 0.2);  // 20% chance rainy

        // Normalize probabilities (optional but ensures sum = 1.0)
        sunny.normalizeTransitions();
        rainy.normalizeTransitions();
        cloudy.normalizeTransitions();

        // Create the Markov chain starting with Sunny
        BreansMrk.Chain weatherChain = new BreansMrk.Chain(sunny);
        weatherChain.addNode(rainy);
        weatherChain.addNode(cloudy);

        // Print transition matrix
        System.out.println("Transition Matrix:");
        weatherChain.printTransitionMatrix();

        // Simulate weather for 10 days
        System.out.println("\nWeather forecast for the next 10 days:");
        for (int i = 0; i < 10; i++) {
            System.out.println("Day " + (i + 1) + ": " + weatherChain.getCurrentNode().getName());
            weatherChain.step(); // Move to next day's weather
        }
    }
}
