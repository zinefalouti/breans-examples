/*
 * Predicting Nvidia Stock Prices Using Breans Linear Regression Model
 * The Dataset was downloaded from https://www.kaggle.com/datasets/datazng/nvidia-historical-market-data-2023-2024-for-ml
 * The Same Dataset "ndvdstock.csv" Was Converted While the Date Column Was Removed into "nvdstock_numeric.csv" with Convert.java
 */

//Import Breans Linear Regression
import com.ml.breans.BreansReg;

public class NvdStockTrain{


    public static void main(String[]args){


        //Loading the Converted CSV Dataset
        BreansReg.CSVLoader loader = new BreansReg.CSVLoader();
        double[][] data = loader.read("nvdstock_numeric.csv");

        //Declaring the Output Column or Columns
        int[] outputCols = {7}; 

        //Splitting the Dataset into X and Y
        BreansReg.DataSplit split = loader.split(data, outputCols);

        //Creating the Object "model" and Training it
        BreansReg.LinReg model = new BreansReg.LinReg();
        model.train(split.X, split.Y);

        // Example input for the Nvidia stock prediction: must match input columns count
        double[][] exampleToPredict = {
            {210.00,12.49,56427600,7.75,0.03658,198.7091623,209.33}
        };
        // The Array of Factors' Order: Open, Range, Volume, Log_Volume, Return_Percentage,3_Day_Avg_AdjClose(Delay), PriorDay_AdjClose

        //Predict
        double[][] prediction = model.predict(exampleToPredict);

        //Display the Result or Results (If You Have Multiple Output Columuns)
        System.out.println("Prediction result:");
        for (int i = 0; i < prediction.length; i++) {
            for (int j = 0; j < prediction[0].length; j++) {
                System.out.printf("%.6f ", prediction[i][j]);
            }
            System.out.println();
        }

    }

}