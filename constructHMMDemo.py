#!/usr/bin/env python2.7

# Author: Ryan Lorig-Roach (please cite)

# Credits:
#   Brian Thornlow - Fast5Types.py
#   Jacob Schreiber - PyPore
#   Kevin Karplus - SpeedyStatSplit (segmenter)
#   Jacob Schreiber and Adam Novak - yahmm (hmm toolkit)

# Models a sequencing construct containing polyT, Barcodes, and adapters. Demonstrates building, training, and plotting.
# See writeup in "Nanopore" repo from rlorigro@github for more info.

# Dependencies:
#   Fast5Types.py
#   PyPore
#   demoNegativeReads (input TSV files)
#   demoPositiveReads (input TSV files)
#   MetaHMMBuilder.py

from Fast5Types import *
import numpy as np
from matplotlib import pyplot as plot
import copy
import os
import yahmm
from kmerAnalysis import *
import random
from MetaHMMBuilder import *

# reference file containing all 6mer DNA means
filename = "kmerMeans"

# Where to find the neg and pos test sets (one file per read, one segment per line, tsv: start end mean std	min max duration)
positiveDirectory = "demoPositiveReads"
falseDirectory = "demoNegativeReads"

builder = metaHMMBuilder(filename)

# INPUT SEQUENCES FOR EACH LINEAR HMM IN THE META-HMM: "Lead" = Y-adapter, "pA" = polyA, "pT" = polyT

fLeadMeans = list(map(float,[125,73,94,76,96,80,81,98,98,65,102,74.6, 62.4, 95.33333333, 99.75, 87.8, 92.75, 77.8]))
fLeadStdevs = list(map(float,[14,4.3,3,6,2.7,5,8,5,5,5,5,1.949358869, 3.361547263, 1.577350269, 1.707825128, 3.033150178, 4.573474245, 1.788854382]))

rLeadMeans = list(map(float, [95.75, 70.66666667, 62.75, 131, 98.75, 62, 80.75, 57.25, 47, 54.75]))
rLeadStdevs = list(map(float, [5.560275773, 3.055050463, 1.258305739, 5.715476066, 2.217355783, 4.082482905, 8.057087977, 3.095695937, 2.645751311, 1.258305739]))

rLeadTailMeans = list(map(float, [62.75, 81.66666667, 93, 77.75, 80]))
rLeadTailStdevs = list(map(float, [2.217355783, 4.041451884, 7.874007874, 44.68687354, 3.559026084]))

hairpinMeans = list(map(float,[103.5, 96.66666667, 69.875, 99.375, 106.875, 84.75, 109, 73.125, 91.75, 97.5, 83.5, 106.625, 73, 88.625, 101.375, 83.75, 97.375, 82.625, 62.75, 77.33333333, 94, 85.5, 94.375, 100.2857143, 75.14285714, 61.85714286, 98.375, 75, 84.125, 88.5, 95.75, 74.28571429, 134.25, 132.625, 129.375, 121.5, 113.375, 85, 74.875, 79.625, 86.25, 87.375, 131.5, 133, 125.875, 119, 110.25, 86.375, 80.375, 75.375, 83.25, 88.4, 99.625, 87.16666667, 106.75, 91.57142857, 77.375, 83.83333333, 104, 86.66666667, 73.33333333, 65.33333333, 84.33333333]))
hairpinStdevs = list(map(float,[5.042675027, 3.076794869, 6.937218463, 6.696214282, 8.42509008, 9.42956344, 5.631543813, 2.90012315, 5.092010549, 5.732115042, 3.271085447, 3.739270364, 2.563479778, 4.405759218, 6.926914382, 3.195979617, 6.412877669, 3.925648263, 5.063877679, 1.751190072, 3.023715784, 2.672612419, 3.020761493, 5.529143566, 6.256425269, 3.287784027, 3.335416016, 3.927922024, 8.626164518, 6.718843438, 1.908627031, 4.644505202, 10.41633333, 10.74293788, 4.897156609, 4.535573676, 6.345695954, 5.126959556, 2.474873734, 5.705573716, 3.412163118, 2.924648941, 5.903993806, 7.0305456, 7.772432971, 5.451081151, 6.541078985, 4.926241685, 4.56500665, 4.068608046, 3.991061441, 2.408318916, 2.924648941, 2.786873995, 4.131758533, 2.149196971, 2.924648941, 3.656045222, 4.242640687, 4.546060566, 3.326659987, 3.265986324, 3.011090611]))

homopolymerLength = 5
pA_means = list(map(float, [83.459321]*homopolymerLength))
pA_stdevs = list(map(float, [2.5]*homopolymerLength))

pT_means = list(map(float, [87.762283]*homopolymerLength))
pT_stdevs = list(map(float, [2.5]*homopolymerLength))

transitions = {"fLead":{"pT":0.4,
                       "pA":0.4,
                       "B1":0.1,
                       "B2":0.1},
               "rLead":{"end":1/2.0,
                        "rLeadTail":1/2.0},
               "hairpin":{"pA":1/3.0,
                          "pT":1/3.0,
                          "B1":1/6.0,
                          "B2":1/6.0},
               "B1":{"B2":0.2,
                     "hairpin":0.1,
                     "pA":0.3,
                     "pT":0.3,
                     "rLead":0.08,
                     "end": 0.02},
               "B2":{"B1":0.2,
                     "hairpin":0.1,
                     "pA":0.3,
                     "pT":0.3,
                     "rLead":0.08,
                     "end": 0.02},
               "start":{"fLead":1},
               "pA":{"hairpin":0.1,
                     "pA":0.4,
                     "B1":0.2,
                     "B2":0.2,
                     "rLead":0.08,
                     "end":0.02},
               "pT": {"hairpin": 0.1,
                      "pT":0.4,
                      "B1": 0.2,
                      "B2": 0.2,
                      "rLead": 0.08,
                      "end": 0.02},
               "rLeadTail":{"end":1}}

nodes = {"B1": "GGGTTCAATCAAGGGTTCAATCAAGGGTTCAATCAAGGGTTCAATCAAT",
         "B2": "TTGATTGAACCCTTGATTGAACCCTTGATTGAACCCTTGATTGAACCCAAAAAA",
         "fLead": (fLeadMeans,fLeadStdevs),
         "rLead": (rLeadMeans, rLeadStdevs),
         "rLeadTail": (rLeadTailMeans,rLeadTailStdevs),
         "pA": (pA_means,pA_stdevs),
         "pT": (pT_means,pT_stdevs),
         "hairpin": (hairpinMeans,hairpinStdevs)}


builder.buildMetaHMM(nodes,transitions)
builder.model.bake(verbose=False) #verbose=True
# builder.sample()

readModel = builder.model
nullModel = Model()

# Build null model to be used for log likelihood ratios (this model is extremely simple, not trained true negatives)
dist = NormalDistribution(100,60)
state = State(dist)
nullModel.add_state(state)
nullModel.add_transition(nullModel.start,state,1)
nullModel.add_transition(state,nullModel.end,0.005)
nullModel.add_transition(state,state,1-0.005)
nullModel.bake(verbose=True)

# Read the files into 2D lists
signalDataSeparated,signalDataBulk = readSegmentFiles(positiveDirectory)
falseSignalDataSeparated,falseSignalDataBulk = readSegmentFiles(falseDirectory)

# extract only means from the files, to be used in the HMM as the sequence
signalMeansSeparated = extractSegmentMeans(signalDataSeparated)
falseSignalMeansSeparated = extractSegmentMeans(falseSignalDataSeparated[:7])

# Shuffle the positive test reads so that each training set is randomly selected
random.shuffle(signalMeansSeparated)
signalMeansSeparated = signalMeansSeparated[:120]

print("Total number of samples: %d"%len(signalMeansSeparated))

# number of test samples
nTest = int(round(len(signalMeansSeparated)*0.333))

for j in range(0,1):
    # Choose (n-nTest) training reads and (nTest) testing reads
    signalsTesting = signalMeansSeparated[nTest*j:nTest*j+nTest]  #j=0
    falseSignalsTesting = falseSignalMeansSeparated
    signalsTraining = signalMeansSeparated[:nTest*j] + signalMeansSeparated[nTest*j+nTest:]

    print("Training Samples: %d" % len(signalsTraining))
    print("Test Samples: %d" % len(signalsTesting))
    print("False Test Samples: %d" % len(falseSignalsTesting))

    # print("log likelihoods before training:")
    #
    # # calculate log likelihoods with the untrained model
    # for s,signal in enumerate(signalsTraining):
    #
    #     nullProb = nullModel.log_probability(signal)
    #     readProb = readModel.log_probability(signal)
    #     logLikelihoodBits = (readProb - nullProb)/math.log(2,math.e)
    #
    #     print("%d: %f" % (s,logLikelihoodBits))


    print("Training...")

    # train the model using the randomly selected training set
    builder.model.train(signalsTraining, stop_threshold=1E-0, min_iterations=0, max_iterations=None, algorithm='viterbi', verbose=True, transition_pseudocount=0, use_pseudocount=False, edge_inertia=0.5) #A wrapper for the 'baum-welch', 'viterbi', and 'labelled' training algorithms. Allows you to specify uniform pseudocounts using transition_pseudocount, use the pseudocounts associated with each edge when building the model, have edge weight inertia, and set a threshold on the emitting characters in Baum-Welch. See the training entry for more details.

    for signalSet in [signalsTesting,falseSignalsTesting]:
        metaStates = set()
        paths = list()

        print("log likelihoods after training:")

        # For each test sequence find the most probable path and extract all the meta-state names (for labelling)
        for s,signal in enumerate(signalSet):
            nullProb = nullModel.log_probability(signal)
            readProb = readModel.log_probability(signal)
            logLikelihoodBits = (readProb-nullProb)/math.log(2, math.e)


            print("%d: %f"%(s, logLikelihoodBits))

            viterbiOutput = builder.model.viterbi(signal)

            path = [(node[1].name).split('_')[-1] for node in viterbiOutput[1]][1:-1]

            # print(" -> ".join(path)) #print the (meta) path

            for i,item in enumerate(path):
                # if item == "pT" or item == "pA": Model.maximum_a_posteriori(signal[i])
                metaStates.add(item)

            paths.append(path) #save the paths for plotting

        metaStates = sorted(list(metaStates))
        # print(metaStates)

        # colors = ["red", "purple", "blue", "orange", "green", "yellow", "brown", "black", "gray"]

        colors = {"B1": [0.004, 0.702, 0.733],
                 "B2": [0.082, 0.282, 0.776],
                 "fLead": "black",
                 "rLead": "black",
                 "rLeadTail": "black",
                 "pA": [0.965, 0, 0.094],
                 "pT": [1.00, 0.671, 0],
                 "hairpin": [0.565, 0.863, 0]}

        kmerCalc = KmerCalculator("kmerMeans")

        print("PLOTTING FIRST 7 ALIGNED SIGNALS OF TEST SET...")
        plotAlignedSignals(signalSet[:7],paths,colors)
