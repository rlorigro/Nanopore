#!/usr/bin/env python2.7

# Author: Ryan Lorig-Roach


import os
from matplotlib import pyplot as plot
from matplotlib import patches as mplPatches
import numpy as np
import sys

class KmerCalculator:
    '''
    Some tools for estimation of a signal from a sequence (if you happen to have a table of means for all kmers...)
    '''
    def __init__(self, file_stdKmerMeans):
        self.k = 6
        self.standardKmerMeans = self.readStandardKmerMeans(file_stdKmerMeans)


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


    def calculateExpectedSignal(self, sequence):
        '''
        Given a sequence, use standard kmer signal values to estimate its expected signal (over all kmers in sequence)
        '''

        signal = list()

        for i in range(0, len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            signal.append(self.standardKmerMeans[kmer])

        return signal


def readSegmentFiles(directory):
    keys = ("start", "end", "mean", "std", "min", "max", "duration") #the column labels for kmer segment data
    kmerDataSeparated = list()
    kmerDataBulk = list()

    for f,filename in enumerate(os.listdir(directory)):
        kmerDataSeparated.append(list())

        with open(directory+'/'+filename,'r') as file:
            file.readline()

            for line in file:
                kmer = list(map(float,line.split('\t')))
                kmerDataBulk.append(kmer)
                kmerDataSeparated[f].append(kmer)

    return kmerDataSeparated,kmerDataBulk


def generateHistogram(xmin,xmax,interval,y):
    '''
    Helper function for plotting histograms
    '''
    # Establish axis with defined limits
    panel = plot.axes()

    bins = np.arange(xmin,xmax,interval)

    frequencies, binsPlaceholder = np.histogram(y, bins)

    for i,frequency in enumerate(frequencies):
        left = bins[i]
        bottom = 0
        width = bins[i+1]-bins[i]
        height = frequency

        rectangle = mplPatches.Rectangle((left, bottom), width, height,
                                          linewidth=0.1,
                                          facecolor=(0.5, 0.5, 0.5),
                                          edgecolor=(0, 0, 0))
        panel.add_patch(rectangle)

    print(frequencies)

    panel.set_xlim([0, xmax])
    panel.set_ylim([0, 1.1*max(frequencies)])

    # Turn off top/right ticks
    panel.tick_params(axis='both', which='both',
                      bottom='on', labelbottom='on',
                      left='on', labelleft='on',
                      right='off', labelright='off',
                      top='off', labeltop='off')

def extractSegmentMeans(kmerDataSeparated):
    '''
    Used in conjunction with the output TSV from plotF5.py to extract a list of segment mean values.
    '''
    signalsSeparated = list()
    for sequence in kmerDataSeparated:
        signalsSeparated.append(np.array(sequence)[:,2])

    return signalsSeparated


def histSegments(kmerMeans):
    '''
    Make a histogram of the segment means values found in a signal
    '''
    plot.figure()

    generateHistogram(0,240,2.5,kmerMeans)

    plot.show()


def plotSegmentedSignals(signalsSeparated):
    nplots = len(signalsSeparated)

    if nplots > 1:
        fig, axes = plot.subplots(nplots, sharex=True, sharey=True)
    else:
        fig = plot.figure(figsize=(24,3))
        axes_s = plot.axes()

    handles = list()
    states = set()
    stateNames = list()

    for s, signal in enumerate(signalsSeparated):
        if nplots > 1:
            axes_s = axes[s]

        if s==0:
            axes_s.set_title("Signal")

        colors = ["red","blue","orange","purple","green","yellow","brown","black",[0.004, 0.702, 0.733]]

        if s >= len(signalsSeparated)-2:
            color = colors[-(s+1)]
        else:
            color = "black"

        # plot.figure()
        # panel = plot.axes()
        # panel.plot(means)

        n = len(signal)
        # length = 100 / float(n)
        length = 5

        for i,kmerMean in enumerate(signal):
            x0 = i*length
            x1 = x0+length
            y = kmerMean

            if i >0:
                xprev = i*length
                yprev = signal[i-1]
                axes_s.plot([xprev,x0],[yprev,y],color=color)

            # print i,n,length,x0,x1,y

            axes_s.plot([x0,x1],[y,y],color=color)
            # axes_s.text(x1, y, "%.3f"%y, ha="right", va="top", color="red", fontsize="5")
            # axes_s.set_ylabel("%d"%s)
            axes_s.set_ylabel("Current (pA)")
            # axes[s].set_ylim([0,150])
            # axes[s].set_yticks(np.linspace(0,150,4))

            axes_s.tick_params(axis='both', which='both',
                          bottom='off', labelbottom='off',
                          left='on', labelleft='on',
                          right='off', labelright='off',
                          top='off', labeltop='off')

    # f.savefig("barcodeComparison2.png",dpi=600)
    plot.show()

