#!/usr/bin/env python2.7

# Author: Ryan Lorig-Roach

import numpy
import random
import sys
import copy
from matplotlib import pyplot
from kmerAnalysis import plotSegmentedSignals

class KmerCalculator:
    '''
    Performs basic functions on DNA sequence. Requires a table of mean current values for all kmers.
    '''
    def __init__(self,kmerMeansFilename):
        self.k = 6
        self.kmerMeans = self.readStandardKmerMeans(kmerMeansFilename)

    def calculateExpectedSignal(self, sequence):
        '''
        Given a sequence, use standard kmer signal values to estimate its expected signal (over all kmers in sequence)
        '''

        expectedSignal = list()
        sequence = ''.join(sequence)

        for i in range(0, len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            expectedSignal.append(self.kmerMeans[kmer])

        return expectedSignal

    def readStandardKmerMeans(self, file_stdKmerMeans):
        '''
        Read a file containing the list of all kmers and their expected signal means (2 columns, with headers)
        '''

        standardKmerMeans = dict()

        with open(file_stdKmerMeans, 'r') as file:
            file.readline()

            for line in file:
                data = line.strip().split()

                if len(data[0])!=self.k:
                    sys.exit("ERROR: kmer length not equal to parameter K")

                try:
                    standardKmerMeans[data[0]] = float(data[1])
                except:
                    print("WARNING: duplicate kmer found in standard means reference list: %s" % data[0])

        return standardKmerMeans


class Seed:
    '''
    Object containing information pertaining to a single seed sequence in SequenceGenerator. Updated each iteration.
    '''
    def __init__(self,sequence,signal=None):
        self.sequence = sequence
        self.signal = signal
        self.error = None
        self.bestSequence = None
        self.bestError = sys.maxsize


class SequenceGenerator:
    '''
    Randomly samples and iterates over sequence space to obtain a sequence with an emission that matches
    the template signal.
    '''
    def __init__(self,kmerMeansFilename):
        self.kmerCalculator = KmerCalculator(kmerMeansFilename)

        self.k = 6
        self.nReseeds = 30
        self.nIterations = 400  # how many single-character alterations to perform for each seed
        self.nSeeds = 100       # how many randomly initialized sequences to start from
        self.maxResults = 4     # the number of saved/plotted results

        self.templateSignal = None  # the desired signal to converge on
        self.s = None
        self.seeds = list()

        self.characters = ['A','C','G','T']

    def generateSequence(self,templateSignal):
        '''
        The master function

        Stage 1: Generate random seeds (nSeeds)
        Stage 2: Sample and substitute all seeds n times (nIterations), then select best result for each seed
        Stage 2: Prune seeds by keeping only the top n seeds (maxResults)
        Stage 3: Refine top seeds by sampling/substituting, then selecting the top match from each seed (nIterations)

        Repeat stage 3 until convergence (nReseeds)... scales by roughly 5x length of the sequence 

        Plot results with the first plot showing the template signal
        '''


        self.templateSignal = templateSignal
        if self.nReseeds == None:
            self.nReseeds = int(5*len(self.templateSignal))

        resultSequences = list()
        updateString = ''

        for rs in range(self.nReseeds):
            if rs == 0:
                # Stage 1
                # on the first iteration, generate random seed sequences
                seedSequences = [[random.choice(self.characters) for i in range(6+len(self.templateSignal)-1)] for s in range(self.nSeeds)]
                for sequence in seedSequences:
                    seedSignal = self.kmerCalculator.calculateExpectedSignal(sequence)
                    self.seeds.append(Seed(copy.deepcopy(sequence),seedSignal))  # initialize Seed object for each random sequence

            elif rs == 1:
                # Stage 3
                # from 2nd iteration on, refine only the best seeds of the first pass
                self.pruneSeeds()

            for s,seed in enumerate(self.seeds):
                # Stage 2/3, refine each seed
                self.s = s

                scorePerIteration = list()  # for troubleshooting (and curiosity)

                for i in range(0, self.nIterations):
                    index = self.selectIndexByError()
                    newCharacter = self.selectSubstitutionByError(index)

                    self.seeds[self.s].sequence[index] = newCharacter
                    self.seeds[self.s].signal = self.kmerCalculator.calculateExpectedSignal(self.seeds[self.s].sequence) # refresh signal

                    self.seeds[self.s].error = self.calculateTotalError()
                    scorePerIteration.append(self.seeds[self.s].error)  # for diagnostic purposes

                    if self.seeds[self.s].error < self.seeds[self.s].bestError:
                        # this seed's best error/sequence is now the current error/sequence
                        self.seeds[self.s].bestSequence = copy.deepcopy(self.seeds[self.s].sequence)
                        self.seeds[self.s].bestError = copy.deepcopy(self.seeds[self.s].error)

                updateString = updateString + "%d %s : %f\n"%(s,''.join(self.seeds[self.s].bestSequence),self.seeds[self.s].bestError)

                if (s+1)%(self.maxResults) == 0 and s>0:
                    if rs >0:
                        print(rs)
                    print(updateString.strip())
                    updateString = ''

                # pyplot.plot(scorePerIteration)
                pyplot.show()

        if rs == (self.nReseeds - 1):
            string = ''
            for s,seed in enumerate(sorted(self.seeds,key = lambda x: x.bestError)):
                string += "%s from seed %s : %f\n"%(''.join(seed.bestSequence),''.join(seed.sequence),seed.bestError)
                # print(seed.signal)
                resultSequences.append(self.kmerCalculator.calculateExpectedSignal(seed.bestSequence))

            print(string)
        plotSegmentedSignals([self.templateSignal]+resultSequences)

    def pruneSeeds(self):
        '''
        Sort the seeds by their best matches and keep only the top n where n = maxResults
        '''

        self.seeds.sort(key = lambda x: x.bestError)
        self.seeds = self.seeds[:self.maxResults]

        for i in range(len(self.seeds)):
            self.seeds[i].sequence = copy.deepcopy(self.seeds[i].bestSequence)
            self.seeds[i].signal = self.kmerCalculator.calculateExpectedSignal(self.seeds[i].sequence)

    def selectSubstitutionByError(self, index):
        '''
        Select a character to substitute at the given index with probability corresponding to the inverse magnitude of
        its error, i.e.: choose better matching characters with greater probability
        '''

        cumulativeSampleSpace = list()

        scoreSum = 0
        for newCharacter in self.characters:
            score = 1.0/self.scoreKernel(index, newCharacter)
            score = score**2

            cumulativeSampleSpace.append((newCharacter, scoreSum))  # start with 0
            scoreSum += score  # add to previous score

        selection = random.random()*scoreSum

        character = None
        for item in reversed(cumulativeSampleSpace):
            if selection > item[1]:  # if this item was selected (with probability proportional to score)
                character = item[0]
                break

        return character

    def selectIndexByError(self):
        '''
        Select an index in the sequence with a probability corresponding to the magnitude of its error, which depends
        on all neighboring kmers that overlap this position.
        '''

        # iterate the full sequence and find the error of each position
        cumulativeSampleSpace = list()

        scoreSum = 0
        for i in range(len(self.templateSignal)):
            cumulativeSampleSpace.append(scoreSum)              # start with 0
            scoreSum += self.scoreKernel(i,self.seeds[self.s].sequence[i])**2    # add to previous score

        selection = random.random()*scoreSum

        index = None

        # choose a position i with probability_i = error_i/(sum(error_j) for j in sequence)
        # by ratcheting through a cumulative sample space up to a random value between 0 and sum(error)
        for i,event in enumerate(reversed(cumulativeSampleSpace)):
            if selection > event:
                index = len(cumulativeSampleSpace) - i - 1
                break

        return index

    def scoreKernel(self,index,character):
        '''
        Assuming a uniform kernel with distance of k-1 characters from the given index, find the error value of
        the expected signal generated by the sequence within the kernel, if the given sequence index is replaced with
        the given character
        '''

        kernelSequenceIndices = [index+i for i in range(-5,6) if index+i>=0 and index+i<len(self.seeds[self.s].sequence)]
        kernelSignalIndices = [index+i for i in range(-5,1) if index+i>=0 and index+i<len(self.seeds[self.s].signal)]

        kernelTemplateSignal = [self.templateSignal[i] for i in kernelSignalIndices]

        kernelSequence = [self.seeds[self.s].sequence[i] if i!=index else character for i in kernelSequenceIndices]
        kernelSignal = self.kmerCalculator.calculateExpectedSignal(kernelSequence)

        error = 0
        for i in range(len(kernelSignalIndices)):
            error += abs(kernelTemplateSignal[i] - kernelSignal[i])

        # return numpy.log2(score)
        return error

    def calculateTotalError(self):
        error = 0
        for i in range(len(self.seeds[self.s].signal)):
            error += abs(self.templateSignal[i] - self.seeds[self.s].signal[i])

        # return numpy.log2(error)
        return error


sequenceGenerator = SequenceGenerator("kmerMeans")

x = numpy.arange(0.0,51.0)
# print(x)

# component1 = (-((1/25.0)*(x-25))**2+1)
# component2 = numpy.sin(0.8*x)*28  #(-(1/25(x-25))^2+1)*sin(0.8*x)*28+80
# signal = list(component1*component2+80)
# signal = list(component1)
# signal = list(numpy.sin(0.5*x)*28+80)
# signal = [69.0,72.0]
# signal = (([100]*2)+([60]*2))*12
signal = [60,70,80,90,100,90,80,70]*2
# signal = [60,70,80,90,100]*2
# signal = [105,90,75,60]*4
# signal = [60,70,80,90,100]*4
# signal = [110,90,75,60,80,95]*4
# signal = ([100,90]*3+[60,70]*3)*2
# signal = [115,85,55,85]*4
# signal = [115,54,115,54]*2
# signal = (([110]*1)+([55]*1)+([110]*1)+([55]*1))*1
# signal = [60,80,100]*2
# signal = list(numpy.arange(60,110,50.0/25.0)) + list(numpy.arange(110,60,-50.0/25.0))

# print(signal)

# plotSegmentedSignals([signal])

sequenceGenerator.generateSequence(signal)
