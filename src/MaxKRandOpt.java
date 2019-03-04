import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.EvaluationFunction;
import opt.DiscreteChangeOneNeighbor;

import opt.example.*;

import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.ga.SingleCrossOver;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;

import shared.FixedIterationTrainer;

public class MaxKRandOpt {

  private static final int N = 50; // number of vertices
  private static final int L =4; // L adjacent nodes per vertex
  private static final int K = 8; // K possible colors

  public static void main(String[] args) {
    System.out.println("Hello World");
    MaxKColoring(N);
    System.out.println("Done");
  }

  public static void MaxKColoring(int N) {

    System.out.println("MAX K COLORING OPTIMZATION PROBLEM:");

    Random random = new Random(N*L);
    // create the random velocity
    Vertex[] vertices = new Vertex[N];
    for (int i = 0; i < N; i++) {
        Vertex vertex = new Vertex();
        vertices[i] = vertex;
        vertex.setAdjMatrixSize(L);
        for(int j = 0; j <L; j++ ){
           vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
        }
    }

    int[] ranges = new int[N];

    Arrays.fill(ranges, 2);

    Distribution perm_dist = new DiscretePermutationDistribution(K);
    Distribution tree_dist = new DiscreteDependencyTree(.1);

    MaxKColorFitnessFunction eval_func = new MaxKColorFitnessFunction(vertices);
    NeighborFunction neigh_func = new SwapNeighbor();
    MutationFunction mut_func = new SwapMutation();
    CrossoverFunction cross_func = new UniformCrossOver();

    HillClimbingProblem hill_climbing = new GenericHillClimbingProblem(eval_func, perm_dist, neigh_func);
    GeneticAlgorithmProblem gen_alg = new GenericGeneticAlgorithmProblem(eval_func, perm_dist, mut_func, cross_func);
    ProbabilisticOptimizationProblem prob_opt = new GenericProbabilisticOptimizationProblem(eval_func, perm_dist, tree_dist);

    long starttime = System.currentTimeMillis();

    FixedIterationTrainer fit;

    System.out.println("RANDOMIZED HILL CLIMBING:");
    int[] numIterationsRHC = {1, 10, 20, 30, 40, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000, 20000, 200000};

    for (int i=0; i < 3; i++) {

      System.out.println("Run: " + i);
      starttime = System.nanoTime();

      for (int iteration: numIterationsRHC) {

        RandomizedHillClimbing rand_hill = new RandomizedHillClimbing(hill_climbing);
        fit = new FixedIterationTrainer(rand_hill, 2000000);
        fit.train();
        System.out.println("RHC: " + eval_func.value(rand_hill.getOptimal()));
        System.out.println(eval_func.foundConflict());
        System.out.println("Time : "+ (System.nanoTime() - starttime));
        System.out.println("============================");
        // System.out.println(eval_func.value(rand_hill.getOptimal()) + " ");
        // System.out.println(((System.nanoTime() - starttime)/Math.pow(10, 9)));

      }
    }
    //
    //
    //
    // System.out.println("SIMULATED ANNEALING:");
    // double[] temps = {1E12, 1E11, 1E10, 1E9, 1E8, 1E7, 1E6, 1E5, 1E4, 1E3, 100.0, 10.0, 1.0};
    //   System.out.println(Arrays.toString(temps));
    //
    // double[] coolExps = {100, 50, 10, 1, 0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.30, 0.20, 0.1, 0.0001};
    // System.out.println(Arrays.toString(coolExps));
    //
    // int[] numIterations = {1, 10, 20, 30, 40, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000, 20000, 200000};
    //
    // for (int i=0; i < 3; i++) {
    //
    //   System.out.println("Run: " + i);
    //   starttime = System.nanoTime();
    //
    //   for (int iterations: numIterations) {
    //
    //     SimulatedAnnealing sim_anneal = new SimulatedAnnealing(1E8, .99, hill_climbing);
    //     fit = new FixedIterationTrainer(sim_anneal, iterations);
    //     fit.train();
    //     System.out.println("SA: " + eval_func.value(sim_anneal.getOptimal()));
    //     System.out.println(eval_func.foundConflict());
    //     System.out.println("Time : "+ (System.nanoTime() - starttime));
    //     System.out.println("============================");
    //     // System.out.println(eval_func.value(sim_anneal.getOptimal()) + " ");
    //     // System.out.println(((System.nanoTime() - starttime)/Math.pow(10, 9)));
    //
    //   }
    //
    // }
    //
    //
    // System.out.println("GENETIC ALGORITHM:");
    // int[] popSizes = {1, 5, 10, 20, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 10000};
    // System.out.println(Arrays.toString(popSizes));
    //
    // int[] mateNums = {10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    // System.out.println(Arrays.toString(mateNums));
    //
    // int[] mutateNums = {0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    // System.out.println(Arrays.toString(mutateNums));
    //
    // int[] numIterationsGA = {1, 10, 20, 30, 40, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000};
    // System.out.println(Arrays.toString(numIterationsGA));
    //
    // for (int i=0; i < 3; i++) {
    //
    //   System.out.println("Run: " + i);
    //   starttime = System.nanoTime();
    //
    //   for (int iterations: numIterationsGA) {
    //
    //     StandardGeneticAlgorithm stand_gen =new StandardGeneticAlgorithm(100, 100/2, 100/20, gen_alg);
    //     fit = new FixedIterationTrainer(stand_gen, iterations);
    //     fit.train();
    //     System.out.println("GA: " + eval_func.value(stand_gen.getOptimal()));
    //     System.out.println(eval_func.foundConflict());
    //     System.out.println("Time : "+ (System.nanoTime() - starttime));
    //     System.out.println("============================");
    //     // System.out.println(eval_func.value(stand_gen.getOptimal()) + " ");
    //     // System.out.println(((System.nanoTime() - starttime)/Math.pow(10, 9)));
    //
    //   }
    //
    // }



    System.out.println("MIMIC");

    int[] samples = {10, 20, 40, 50, 100, 150, 200, 400, 500, 750, 1000, 2500};
    System.out.println(Arrays.toString(samples));

    int[] toKeeps = {190, 150, 100, 70, 50, 40, 30, 20, 10, 5, 1};
    System.out.println(Arrays.toString(toKeeps));

    int[] numIterationsMM = {1, 10, 20, 30, 40, 50, 100, 200, 500, 700, 1000, 2000, 5000, 10000};

    for (int i=0; i < 3; i++) {

      System.out.println("Run: " + i);
      starttime = System.nanoTime();

      for (int iteration: numIterationsMM) {

        MIMIC mimic = new MIMIC(200, 200/5, prob_opt);
        fit = new FixedIterationTrainer(mimic, 200000);
        fit.train();
        System.out.println("MIMIC: " + eval_func.value(mimic.getOptimal()));
        System.out.println(eval_func.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        System.out.println("============================");
        System.out.println(eval_func.value(mimic.getOptimal()) + " ");
        System.out.println(((System.nanoTime() - starttime)/Math.pow(10, 9)));
      }

    }

  }
}
