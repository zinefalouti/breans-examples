/*
 * Example of a Banana Quality Prediction Using Breans Neural Network for Classification
 */


import java.util.Arrays;

//Importing Breans
import com.ml.breans.Breans;

public class BananaPredict {
    

    public static void main(String[]args) throws Exception{


        //Let's Load the Model We Saved in BananaTrain
        Breans.LoadedModel loaded = Breans.loadNetwork("banana-model.txt");

        //Let's Declare the Factors of an Example Banana So We Can Predict Its Quality
        double[] Banana = new double[] {-5.76, 1.62, 3.62, -1.7133021, -2.96, 4.07941, 3.2912027};

        // Another Banana Example //
        // double[] Banana = new double[] {-4.1676183,-0.9377794,-0.31852946,-1.1211845,-6.567659,-1.1257954,-0.26305485};
        //The Array Includes the 7 Factors in the Dataset Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity

        //Let's Predict Using the LoadedModel Object "loaded" the Quality of the Banana Array
        System.out.println("Model loaded. Predict: " +
        Arrays.toString(loaded.predict(Banana)));

        double[] prediction = loaded.predict(Banana);

        /*Classes Are Hot Coded While We Used loadCSVDatasetWithCategorical in Training
         *Let's Reverse That To Print a String While We Interpret Like This:
         [1,0] is a Good Banana
         [0,1] is a Bad Banana
        */

        int predictedClass = prediction[0] > prediction[1] ? 0 : 1;
        String label = predictedClass == 0 ? "Banana is Good" : "Banana is Bad";
        System.out.println("Prediction: " + label + " (Confidence: " + prediction[predictedClass] + ")");


    }

}
