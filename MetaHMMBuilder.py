#!/usr/bin/env python2.7

# Author: Ryan Lorig-Roach (please cite)

# Credits:
#   Brian Thornlow - Fast5Types.py
#   Jacob Schreiber - PyPore
#   Kevin Karplus - SpeedyStatSplit (segmenter)
#   Jacob Schreiber and Adam Novak - yahmm (hmm toolkit)

# Modularly construct large branched HMMs for classifying sequence elements within nanopore signals.
# See "constructHMMDemo" for usage
# See writeup in "Nanopore" repo from rlorigro@github for more info.

# Dependencies:
#   Fast5Types.py
#   PyPore

from Fast5Types import *
import numpy as np
from matplotlib import pyplot
import copy
import os
import yahmm
from kmerAnalysis import *
import random


class metaHMMBuilder:
    def __init__(self,file_standardKmerMeans):
        self.metaNodes = dict()
        self.metaTransitions = dict()
        self.HMMs = dict()
        self.HMM = None
        self.model = Model(name="meta")
        self.startNode = self.model.start
        self.endNode = self.model.end
        self.kmerCalc = KmerCalculator(file_standardKmerMeans)
        self.expectedMeans = None
        self.k = self.kmerCalc.k
        self.sequence = None
        self.sequenceIsString = None
        self.unknownBridgeStdev = 7.5

        # PRESET TRANSITION PROBABILITIES
        # For all non-terminal nodes. Node->end probabilities will be filled in with node->next probabilities.

        self.transitions = {"skip": {"next": 0.01, #unused
                                     "emit": 1.0},

                            "blip": {"emit": 0.5,
                                     "next": 0.5},

                            "drop": {"next": 0.7,
                                     "emit": 0.3},

                            "emit": {"self": 0.015,
                                     "next": 0.953,
                                     "prev": 0.0165,
                                     "skip": 0.01,
                                     "blip": 0.005,
                                     "drop": 0.0005},

                            "start": {"emit": 0.982,
                                      "skip": 0.015,
                                      "drop": 0.0025,
                                      "blip": 0.0005}}

        # quick check to determine whether presets sum to 1. Either way, the yahmm package normalizes transitions later
        for key1 in self.transitions:
            totalProbability = float(sum([self.transitions[key1][key2] for key2 in self.transitions[key1]]))

            if totalProbability != 1.0: # still don't know why this inequality fails for 1.0==1.0 ...
                print("WARNING: Total outgoing transition probability for %s nodes != 1.0, p = %f"%(key1,totalProbability))

    def newPhase(self,skipNode=None,blipNode=None,dropNode=None,emitNode=None):
        '''
        Generate a container to be used for each phase in the linear HMM
        '''

        # Each slot is initialized as None unless otherwise specified
        phase =  {"skip": skipNode,
                  "blip": blipNode,
                  "drop": dropNode,
                  "emit": emitNode}

        return phase

    def buildMetaHMM(self,nodes,transitions):
        '''
        Given a set of sequences (meta-nodes) and a set of transitions between them, build an HMM.

        nodes is a dict of the format {"name":sequence} where sequence is either:
                -DNA sequence (string)
                -list of means (float)
                -a tuple of means (float list) and stdevs (float list)

        transitions is a dict of dict w/ transition probabilities. e.g.: {"nodeName1":{"nodeName2":1.0}}
        '''

        self.metaNodes = nodes
        self.metaTransitions = transitions

        for nodeName in self.metaNodes:  # !!! need error proofing for DNA sequence input, must be A,C,G,T only !!!
            if nodeName != "start":

                data = nodes[nodeName]

                # if a nucleotide sequence or list of means is provided
                if type(data) == str or type(data) == list:
                    sequence = data

                    self.buildDraftHMM(nodeName,sequence)

                # if a list of stdevs is *also* provided within a tuple
                if type(data) == tuple:
                    # print(len(data[0]))
                    # print(len(data[1]))
                    self.buildDraftHMM(nodeName,data[0],stdevs=data[1])


        for nodeName in self.metaTransitions:
            if nodeName == "start":
                for nodeName2 in self.metaTransitions["start"]:
                    self.addStartTransitions(self.HMMs[nodeName2],self.metaTransitions["start"][nodeName2])
            else:
                for nodeName2 in self.metaTransitions[nodeName]:
                    if nodeName2 == "end":
                        self.addEndTransitions(self.HMMs[nodeName],self.metaTransitions[nodeName][nodeName2])
                    else:
                        self.bridgeHMMs(self.HMMs[nodeName],self.HMMs[nodeName2],self.metaTransitions[nodeName][nodeName2])

    class linearHMM:
        '''
        Container for storing HMM fragments. This class exists so that metadata associated with linear HMMs can be
        accessed later (e.g. the underlying sequence). LinearHMMs are bridged and spliced to form the meta-HMM.
        '''

        def __init__(self, name, HMM, expectedMeans, sequence, sequenceIsString):
            self.name = name
            self.HMM = HMM
            self.expectedMeans = expectedMeans
            self.sequence = sequence
            self.sequenceIsString = sequenceIsString

    def storeHMM(self,name):
        '''
        Add a linearHMM instance to the dictionary of stored linearHMMs
        '''

        self.HMMs[name] = self.linearHMM(name,self.HMM,self.expectedMeans,self.sequence,self.sequenceIsString)

        self.clearHMM()

    def clearHMM(self):
        '''
        Erase all data pertinent to the current HMM under construction (in preparation for the next)
        '''

        self.HMM = None
        self.expectedMeans = None
        self.sequence = None
        self.sequenceIsString = None

    def bridgeHMMs(self,HMM1,HMM2,transitionProbability):
        '''
        For two linear HMMs, build an HMM between them using the underlying sequence that each input HMM represents.

        HMM1 and HMM2 are instances of the subclass linearHMM, which retain the metadata necessary for bridging
        '''

        bridgeName = "%s-%s"%(HMM1.name,HMM2.name)

        if HMM1.sequenceIsString and HMM2.sequenceIsString:     # both HMMs have a known sequence
            bridgeSeq = self.bridgeSequences(HMM1,HMM2)
            self.buildDraftHMM(bridgeName, bridgeSeq, reverseEmitAllowed=True)
            # print("SPLICING 2 NT SEQS")

        else:
            bridgeSeq = self.bridgeMeans(HMM1,HMM2)         # one or both sequences are a list of means (not a string)
            self.buildDraftHMM(bridgeName, bridgeSeq,defaultEmitStdev=float(self.unknownBridgeStdev), reverseEmitAllowed=True)
            # print("SPLICING 2 MEAN SEQS")

        self.spliceHMMs(HMM1,self.HMMs[bridgeName],transitionProbability,reverseEmitAllowed=False)
        self.spliceHMMs(self.HMMs[bridgeName],HMM2,1,reverseEmitAllowed=False)

    def bridgeMeans(self,HMM1,HMM2):
        '''
        Generate the expected emissions when joining 2 sequences. For the nanopore, k=6 so the bridge is length 5.
        '''

        if HMM1.sequenceIsString:
            seq = HMM1.sequence[-(self.k):]
            startMean = self.kmerCalc.calculateExpectedSignal(seq)[0]
            endMean = HMM2.expectedMeans[0]

        elif HMM2.sequenceIsString:
            seq = HMM2.sequence[:self.k]
            endMean = self.kmerCalc.calculateExpectedSignal(seq)[0]
            startMean = HMM1.expectedMeans[len(HMM1.expectedMeans)-1]
        else:
            startMean = HMM1.expectedMeans[len(HMM1.expectedMeans)-1]
            endMean = HMM2.expectedMeans[0]


        interval = (endMean - startMean)/6.0
        bridgeMeanList = [startMean+interval*(i+1) for i in range(5)]

        return bridgeMeanList

    def bridgeSequences(self,HMM1,HMM2):
        '''
        Find the sequence that arises from joining 2 nucleotide sequences, such that k-1 new kmers are contained within.
        This method is only used when two stored linearHMMs have known nucleotide sequences.
        '''
        seq1 = self.HMMs[HMM1.name].sequence
        seq2 = self.HMMs[HMM2.name].sequence

        bridgeSequence = seq1[-(self.k-1):] + seq2[:self.k-1]

        return bridgeSequence

    def addStartTransitions(self,HMM,transitionProbability):
        inPhase = HMM.HMM[0]

        # # start -> skip
        self.model.add_transition(self.startNode, inPhase["skip"], transitionProbability*self.transitions["start"]["skip"])

        # start -> emit
        self.model.add_transition(self.startNode, inPhase["emit"], transitionProbability*self.transitions["start"]["emit"])

        # start -> blip
        self.model.add_transition(self.startNode, inPhase["blip"], transitionProbability*self.transitions["start"]["blip"])

        # start -> drop
        self.model.add_transition(self.startNode, inPhase["drop"], transitionProbability*self.transitions["start"]["drop"])

    def addEndTransitions(self,HMM,transitionProbability):
        outPhase = HMM.HMM[len(HMM.HMM)-1]

        # # skip -> end
        self.model.add_transition(outPhase["skip"], self.endNode, transitionProbability*self.transitions["skip"]["next"])

        # emit -> end
        self.model.add_transition(outPhase["emit"], self.endNode, transitionProbability*self.transitions["emit"]["next"])

        # drop -> end
        self.model.add_transition(outPhase["drop"], self.endNode, transitionProbability*self.transitions["skip"]["next"])

        # blip -> end
        self.model.add_transition(outPhase["blip"], self.endNode, transitionProbability*self.transitions["blip"]["next"])

    def spliceHMMs(self,startHMM,endHMM,transitionProbability, reverseEmitAllowed=True):
        '''
        Add transitions between the last phase of a linear HMM to the first phase of another. The connections made here
        are the same as the inner loop of buildDraftHMM.
        '''

        phaseA = startHMM.HMM[len(startHMM.HMM)-1]
        phaseB = endHMM.HMM[0]

        # if n > 0:
        if reverseEmitAllowed:
            # emit -> emit (prev) .
            self.model.add_transition(phaseB["emit"], phaseA["emit"], transitionProbability*self.transitions["emit"]["prev"])

        # emit (prev) -> emit .
        self.model.add_transition(phaseA["emit"], phaseB["emit"], transitionProbability*self.transitions["emit"]["next"])

        # blip (prev) -> blip .
        self.model.add_transition(phaseA["blip"], phaseB["blip"], transitionProbability*self.transitions["blip"]["next"])

        # emit (prev) -> skip .
        self.model.add_transition(phaseA["emit"], phaseB["skip"], transitionProbability*self.transitions["emit"]["skip"])

        # skip (prev) -> skip .
        # self.model.add_transition(phaseA["skip"], phaseB["skip"], transitionProbability*self.transitions["skip"]["next"])

        # skip (prev) -> emit .
        self.model.add_transition(phaseA["skip"], phaseB["emit"], transitionProbability*self.transitions["skip"]["emit"])

        # drop (prev) -> drop .
        self.model.add_transition(phaseA["drop"], phaseB["drop"], transitionProbability*self.transitions["drop"]["next"])


    def buildDraftHMM(self,name,sequence,defaultEmitStdev=5.0,stdevs=None,reverseEmitAllowed = True):
        '''
        Build a linear HMM fragment based on user-provided sequence
        '''

        self.sequence = sequence

        if type(sequence) == str:
            self.expectedMeans = self.kmerCalc.calculateExpectedSignal(sequence)
            self.sequenceIsString = True

        elif type(sequence) == list:
            if type(sequence[0]) == float:
                self.expectedMeans = sequence
                self.sequenceIsString = False
            else:
                sys.exit("ERROR: list of means provided not in float format")
        else:
            sys.exit("ERROR: Sequence provided is not nucleotide or list of means")


        self.HMM = list()
        nPhases = len(self.expectedMeans)

        for n in range(nPhases):
            # print(n)
            self.HMM.append(self.newPhase())        # add template (phase) to the HMM to contain each state

            if stdevs != None:
                emitNodeStdev = stdevs[n]           # if stdevs were specified by user, use them.
            else:
                emitNodeStdev = defaultEmitStdev    # if not, use default stdevs (with the assumption they'll be trained)

            # generate the emission distributions for each emitting node in the current phase
            emission = NormalDistribution(self.expectedMeans[n],emitNodeStdev)
            blipEmission = NormalDistribution(self.expectedMeans[n]+10,3)
            dropEmission = NormalDistribution(10,3)

            # generate all the states for current phase, and add them to the phase container for later reference
            self.HMM[n]["skip"] = State(None, "skip_%d_%s" % (n,name))    #silent
            self.HMM[n]["blip"] = State(blipEmission, "blip_%d_%s" % (n,name))
            self.HMM[n]["drop"] = State(dropEmission, "drop_%d_%s" % (n,name))
            self.HMM[n]["emit"] = State(emission, "emit_%d_%s" % (n,name))

            # BUILD ALL INTRA-PHASE TRANSITIONS:
            # emit -> self
            self.model.add_transition(self.HMM[n]["emit"], self.HMM[n]["emit"], self.transitions["emit"]["self"])

            # emit -> blip
            self.model.add_transition(self.HMM[n]["emit"], self.HMM[n]["blip"], self.transitions["emit"]["blip"])

            # emit -> drop
            self.model.add_transition(self.HMM[n]["emit"], self.HMM[n]["drop"], self.transitions["emit"]["drop"])

            # blip -> emit
            self.model.add_transition(self.HMM[n]["blip"], self.HMM[n]["emit"], self.transitions["blip"]["emit"])

            # drop -> emit
            self.model.add_transition(self.HMM[n]["drop"], self.HMM[n]["emit"], self.transitions["drop"]["emit"])

            # BUILD ALL INTER-PHASE TRANSITIONS:
            if n > 0:
                if reverseEmitAllowed:
                    # emit -> emit (prev)
                    self.model.add_transition(self.HMM[n]["emit"],self.HMM[n-1]["emit"],self.transitions["emit"]["prev"])

                # emit (prev) -> emit
                self.model.add_transition(self.HMM[n-1]["emit"],self.HMM[n]["emit"],self.transitions["emit"]["next"])

                # blip (prev) -> blip
                self.model.add_transition(self.HMM[n-1]["blip"],self.HMM[n]["blip"],self.transitions["blip"]["next"])

                # emit (prev) -> skip
                self.model.add_transition(self.HMM[n-1]["emit"],self.HMM[n]["skip"],self.transitions["emit"]["skip"])

                # skip (prev) -> skip
                # self.model.add_transition(self.HMM[n-1]["skip"],self.HMM[n]["skip"],self.transitions["skip"]["next"])

                # skip (prev) -> emit
                self.model.add_transition(self.HMM[n-1]["skip"],self.HMM[n]["emit"],self.transitions["skip"]["emit"])

                # drop (prev) -> drop
                self.model.add_transition(self.HMM[n-1]["drop"],self.HMM[n]["drop"],self.transitions["drop"]["next"])

        self.storeHMM(name)
        # self.printHMM(name)


    def sample(self):
        '''
        Sample the completed model several times and plot the results. Model must be baked.
        '''

        nplots = 4
        f, axes = pyplot.subplots(nplots, sharex=True, sharey=True)

        # signals = [HMM.expectedMeans for HMM in self.HMMs]
        signals = list()
        metaStates = set()
        paths = list()

        for i in range(nplots):
            signal,pathOutput = self.model.sample(path=True)
            signals.append(signal)
            print
            path = [(node.name).split('_')[-1] for node in pathOutput][1:-1]
            print(" -> ".join(path))

            for item in path:
                metaStates.add(item)

            paths.append(path)

        metaStates = sorted(list(metaStates))
        colors = {"B1": "blue",
                  "B2": "purple",
                  "fLead": "black",
                  "rLead": "brown",
                  "rLeadTail": "yellow",
                  "pA": "red",
                  "pT": "orange",
                  "hairpin": "green"}

        for s, signal in enumerate(signals):

            n = len(signal)

            try:
                length = 100 / float(n)
            except ZeroDivisionError:
                length = 0

            for i, kmerMean in enumerate(signal):
                x0 = (i) * length
                x1 = x0 + length
                y = kmerMean

                try: color = colors[paths[s][i]]
                except KeyError: color = "gray"


                if i > 0:
                    xprev = i * length
                    yprev = signal[i-1]
                    axes[s].plot([xprev, x0], [yprev, y], color=color)

                axes[s].plot([x0, x1], [y, y], color=color)
                if s == 3: axes[s].set_ylabel("Current (pA)")
                # axes[s].set_ylim([0,150])
                # axes[s].set_yticks(np.linspace(0,150,4))

                axes[s].tick_params(axis='both', which='both',
                                    bottom='off', labelbottom='off',
                                    left='on', labelleft='on',
                                    right='off', labelright='off',
                                    top='off', labeltop='off')

        axes[0].set_title("Simulated Signals")
        pyplot.show()

    def printHMM(self,name):
        '''
        For a linearHMM, print all the phases and their constituent states.
        '''
        for p,phase in enumerate(self.HMMs[name].HMM):
            print("PHASE%d - "%p),
            for slot in phase:
                try:
                    print("%s"%(phase[slot].name)),
                except:
                    print("%s"%("None")),
            print


#------------------------------------------------------------------------------------------------------


def plotAlignedSignals(signals, paths, colors):
    '''
    Plot and color code an observed signal based on its alignment through an HMM.

    signals: a list of signals in which each signal is a list of mean values (float)
    paths: a list of names (strings) identifying sequence elements one segment at a time, e.g. "lead" (AKA Y-Adapter),
        "bridge", "polyT", "polyA", etc, which correspond to the names of each linearHMM that went into building
        the metaHMM. This is produced by parsing the output of the yahmm.model.viterbi() function
    colors: a dictionary indicating which element names correspond to which color. Keys must be the same as path names.

    Default nplots = all signals, edit below to plot a subset of the signals.
    '''

    nplots = len(signals)

    if nplots > 1:
        fig, axes = pyplot.subplots(nplots, sharex=True, sharey=True)
    else:
        fig = pyplot.figure(figsize=(24, 3))
        axes_s = pyplot.axes()

    handles = list()
    states = set()
    stateNames = list()

    for s, signal in enumerate(signals):
        if nplots > 1:
            axes_s = axes[s]

        if s==0:
            axes_s.set_title("Aligned Observed Signal")

        n = len(signal)
        length = 100.0/float(n)

        axes_s.set_xlim([0,100])

        for i, segmentMean in enumerate(signal):
            x0 = i*length
            x1 = x0+length
            y = segmentMean

            try:
                color = colors[paths[s][i]]
            except KeyError:
                color = "gray"

            if i > 0:
                xprev = i*length
                yprev = signal[i-1]

                axes_s.plot([xprev, x0], [yprev, y], color=color,lw=0.8)

            # print i,n,length,x0,x1,y

            handle, = axes_s.plot([x0, x1], [y, y], color=color,lw=0.8)

            stateName = paths[s][i]
            if '-' in stateName:
                stateName = "Bridge"

            if stateName not in states:
                states.add(stateName)
                stateNames.append(stateName)
                handles.append(handle)

            if i==len(signal)-1:
                box = axes_s.get_position()
                axes_s.set_position([box.x0, box.y0, box.width*0.9, box.height])
                if s == 0:
                    pyplot.legend(handles, stateNames, loc='upper left', bbox_to_anchor=(1, 1))

            if s == 3: axes_s.set_ylabel("Current (pA)")
            # axes[s].set_ylim([0,150])
            # axes[s].set_yticks(np.linspace(0,150,4))

            axes_s.tick_params(axis='both', which='both',
                                bottom='off', labelbottom='off',
                                left='on', labelleft='on',
                                right='off', labelright='off',
                                top='off', labeltop='off')


    # pyplot.savefig("barcodeComparison2.png",dpi=600)
    # pyplot.savefig("alignedReads.pdf")             #save fig as pdf
    pyplot.show()