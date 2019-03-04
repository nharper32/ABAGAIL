import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import src.dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import src.dist.Distribution as Distribution
import src.opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import src.opt.EvaluationFunction as EvaluationFunction
import src.opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import src.opt.HillClimbingProblem as HillClimbingProblem
import src.opt.NeighborFunction as NeighborFunction
import src.opt.RandomizedHillClimbing as RandomizedHillClimbing
import src.opt.SimulatedAnnealing as SimulatedAnnealing
import src.opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import src.opt.ga.CrossoverFunction as CrossoverFunction
import src.opt.ga.SingleCrossOver as SingleCrossOver
import src.opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import src.opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import src.opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import src.opt.ga.MutationFunction as MutationFunction
import src.opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import src.opt.ga.UniformCrossOver as UniformCrossOver
import src.opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import src.opt.prob.MIMIC as MIMIC
import src.opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import src.shared.FixedIterationTrainer as FixedIterationTrainer

from array import array



"""
Commandline parameter(s):
   none
"""

N=200
T=N/5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

rhc = RandomizedHillClimbing(hcp)
fit = FixedIterationTrainer(rhc, 200000)
fit.train()
print "RHC: " + str(ef.value(rhc.getOptimal()))

sa = SimulatedAnnealing(1E11, .95, hcp)
fit = FixedIterationTrainer(sa, 200000)
fit.train()
print "SA: " + str(ef.value(sa.getOptimal()))

ga = StandardGeneticAlgorithm(200, 100, 10, gap)
fit = FixedIterationTrainer(ga, 1000)
fit.train()
print "GA: " + str(ef.value(ga.getOptimal()))

mimic = MIMIC(200, 20, pop)
fit = FixedIterationTrainer(mimic, 1000)
fit.train()
print "MIMIC: " + str(ef.value(mimic.getOptimal()))
