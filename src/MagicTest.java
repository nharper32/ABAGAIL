import dist.*;
import opt.*;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.*;
import shared.Instance;
import shared.ErrorMeasure;
import shared.SumOfSquaresError;
import shared.DataSet;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BackPropagationNetwork;

import java.util.Scanner;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying MAGIC GAMMA Telescope particles as
 * GAMMA or HADRON
 *
 * @author Noah Harper
 * @version 1.0
 */

public class MagicTest {

  //10, 20, 50, 100, 250, 500
  private static int inputLayer = 10, hiddenLayerA = 40, outputLayer = 1, trainingIterations = 10;
  // private static int[] ;

  private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
  private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];

  private static NeuralNetworkOptimizationProblem[] nnProbs = new NeuralNetworkOptimizationProblem[3];
  private static OptimizationAlgorithm[] optAlgs = new OptimizationAlgorithm[3];
  private static String[] optAlgsNames = {"RHC", "SA", "GA"};

  private static Instance[] instances = initializeInstances();
  private static DataSet set = new DataSet(instances);
  private static ErrorMeasure errorMeasure = new SumOfSquaresError();
  private static String results = "";

  private static DecimalFormat decFormat = new DecimalFormat("0.000");




  public static void main(String[] args) {

    int[] iterations_list = {1000};

    for (int iterations: iterations_list) {

      trainingIterations = iterations;
      run();

    }

  }

  private static void run() {

    for (int i = 0; i < optAlgs.length; i++) {

      networks[i] = factory.createClassificationNetwork(
        new int[] {inputLayer, hiddenLayerA, outputLayer}
      );

      nnProbs[i] = new NeuralNetworkOptimizationProblem(set, networks[i], errorMeasure);

    }

    optAlgs[0] = new RandomizedHillClimbing(nnProbs[0]);
    optAlgs[1] = new SimulatedAnnealing(1E11, .95, nnProbs[1]);
    optAlgs[2] = new StandardGeneticAlgorithm(200, 100, 10, nnProbs[2]);

    for (int i = 0; i < optAlgs.length; i++) {

      double start = System.nanoTime();
      double end;
      double trainingTime;
      double testingTime;
      double correct = 0;
      double incorrect = 0;

      System.out.println("\nSTART TRAINING: " + optAlgsNames[i] + "\n--------------------------------------");

      train(optAlgs[i], networks[i], optAlgsNames[i]);

      end = System.nanoTime();
      trainingTime = end - start;
      trainingTime /= Math.pow(10, 9);

      Instance optimalInstance = optAlgs[i].getOptimal();

      networks[i].setWeights(optimalInstance.getData());

      double predicted, actual;
      start = System.nanoTime();

      for (int j = 0; j < instances.length; j++) {

        networks[i].setInputValues(instances[j].getData());
        networks[i].run();

        predicted = Double.parseDouble(instances[j].getLabel().toString());
        actual = Double.parseDouble(networks[i].getOutputValues().toString());

        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(10, 9);

      results += "\nResults for " + optAlgsNames[i] + ": \nCorrectly classified " + correct + " instances." +
                  "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                  + decFormat.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + decFormat.format(trainingTime)
                  + " seconds\nTesting time: " + decFormat.format(testingTime) + " seconds\n";
    }

    System.out.println(results);
  }

  private static void train(OptimizationAlgorithm optAlg, BackPropagationNetwork network, String optAlgName) {

    // System.out.println("\nERROR RESULTS: " + optAlgName + "\n--------------------------------------");

    System.out.println("ITERATIONS: " + trainingIterations);

    for (int i = 0; i < trainingIterations; i++) {

      optAlg.train();

      double error = 0;

      for(int j = 0; j < instances.length; j++) {

        network.setInputValues(instances[j].getData());
        network.run();

        Instance output = instances[j].getLabel();
        Instance example = new Instance(network.getOutputValues());

        example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
        error += errorMeasure.value(output, example);
      }

    }
  }




  private static Instance[] initializeInstances() {

    double[][][] attributes = new double[19020][][];

    try {

      BufferedReader buffer_reader = new BufferedReader(new FileReader(new File("./magic.txt")));

      for (int i = 0; i < attributes.length; i++) {

        Scanner scanner = new Scanner(buffer_reader.readLine());
        scanner.useDelimiter(",");

        attributes[i] = new double[2][];
        attributes[i][0] = new double[10];
        attributes[i][1] = new double[1];

        for (int j = 0; j < 10; j++) {
          attributes[i][0][j] = Double.parseDouble(scanner.next());

        }
        attributes[i][1][0] = Double.parseDouble(scanner.next());
      }
    } catch(Exception e) {

      e.printStackTrace();

    }

    Instance[] instances = new Instance[attributes.length];
    for (int i = 0; i < instances.length; i++) {

      instances[i] = new Instance(attributes[i][0]);
      instances[i].setLabel(new Instance(attributes[i][1][0]));

    }

    return instances;
  }
}
