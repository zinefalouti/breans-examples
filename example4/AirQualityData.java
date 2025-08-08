import com.ml.breans.DataAnalyzer;
import java.io.IOException;

/*
 * This example uses the Air Quality, Weather, and Respiratory Health from Kaggle
 * Link to the dataset: https://www.kaggle.com/datasets/khushikyad001/air-quality-weather-and-respiratory-health
 */

public class AirQualityData{


    public static void main(String[]args) throws IOException{
           
           //CSV to Matrix
           String[][] grid = DataAnalyzer.readCsvToMatrix("air-quality.csv");

           //Check the Shape
           DataAnalyzer.DataShape(grid);

           //Head and Tail
           DataAnalyzer.DataHead(grid, 6);     
           
           DataAnalyzer.DataTail(grid, 6);

           //Check for Blanks or NaN
           DataAnalyzer.FindBlank(grid);

           //Describe the Columns
           DataAnalyzer.Describe(grid);

           //Export and preview the summary in html
           DataAnalyzer.DataSetSummary(grid, "AirQualityDataReport", 6, 6);


           //Creating the object dataset for encoding and manipulation
           DataAnalyzer.DataSet dataset = new DataAnalyzer.DataSet(grid);

            // Hotcode 2nd column (index 1) To make South Central etc numerical
            dataset.EncodeCol(1);

            // Save to a new CSV file
            dataset.saveToCsv("air-quality-modified.csv");
    }

}