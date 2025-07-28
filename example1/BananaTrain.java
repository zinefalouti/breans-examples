/*
 * Example of a Banana Quality Prediction Using Breans Neural Network for Classification
 * Dataset in this example is downloaded from https://www.kaggle.com/datasets/l3llff/banana
*/

//Importing Breans
import com.ml.breans.Breans;

public class BananaTrain {


    public static void main(String[]args) throws Exception{

        //loading CSVs with string labels, so it automatically converts them to one-hot encoded targets
        Breans.Dataset dataset = Breans.loadCSVDatasetWithCategorical("banana_quality.csv", 7);

        //Normalize the Dataset Object We Just Created
        Breans.Normalization norm = Breans.normalizeDataset(dataset);

        /* Designing a Neural Network with 2 Hidden Layers Each with 16 Neurons and All Using 
        a Sigmoid Activation Function and the Output a Softmax */

        Breans.LayerSpec[] networkdesign = {
            new Breans.LayerSpec(16, "sigmoid"),
            new Breans.LayerSpec(16, "sigmoid"),
            new Breans.LayerSpec(dataset.targets[0].length, "softmax")
        };

        //Let's Create The Network We Just Designed
        Breans.Layer[] network = Breans.createNetwork(dataset.inputs[0].length, networkdesign);

        //Setting the Total Epochs and Learning Rate
        int epochs = 60;
        double learningRate = 0.0001;

        //Training Using SGD
        Breans.train(network, dataset, epochs, learningRate);


        //Evaluating To Check Accuracy and Loss
        Breans.evaluateDataset(network, dataset, true);


        //Saving the Model as banana-model.txt
        Breans.saveNetwork(network, "banana-model.txt", norm);

        //The Moment You're Satisfied With the Training Session You Can Save the Model and Use Only banana-model.txt Anywhere With Breans
    }

}


