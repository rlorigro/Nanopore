#!/usr/bin/env python2.7

# Author: Ryan Lorig-Roach

# Credits:
#   Brian Thornlow - Fast5Types.py
#   Jacob Schreiber - PyPore
#   Kevin Karplus - SpeedyStatSplit (segmenter)
#   Jacob Schreiber and Adam Novak - yahmm (hmm toolkit)

# This script parses, segments, and plots fast5 files, optionally binning them based on user input and saving a TSV.
# The TSV of segment data is then used for constructing HMMs.


from Fast5Types import *
import h5py
import numpy
from matplotlib import pyplot as plot
import copy
import os

def parseF5(filename):
    '''
    Adapted from Brian Thornlow's work. Returns an object of the type Fast5FileSet, which contains a single event.
    '''

    f5File = h5py.File(filename,'r')

    # File will have either Raw read or Events
    try:
        reads = (f5File['/Raw/Reads/'])
        readID = reads.keys()[0]
        read = reads[readID]

        signal = (read['Signal']).value

    except:
        print("NO RAW READS")
        read = (f5File['/Analyses/EventDetection_000/Reads/'])
        signal = (read['Events']).value

    # Find values to be used for converting current to picoAmperes
    # Create current - numpy array of floats in units pA
    uniqueKey = (f5File['/UniqueGlobalKey'])
    digitisation = (uniqueKey['channel_id']).attrs['digitisation']
    offset = (uniqueKey['channel_id']).attrs['offset']
    f5range = (uniqueKey['channel_id']).attrs['range']
    sampling_rate = (uniqueKey['channel_id']).attrs['sampling_rate']

    adjusted_signal = (signal + offset) * (f5range / digitisation)
    current = numpy.array(adjusted_signal, dtype=numpy.float)

    # timestep was fADCSequenceInterval * 1e-3 = .01 for .abf
    # Different for .fast5? standard or make use of sampling interval?
    # This is fed into Segmenter as second = 1000/timestep
    timestep = 0.05
    # timestep = 1/sampling_rate
    # ??? sampling_rate = 3012 so would be 0.000332
    # looks weirdly smooth using that, horizontal axis VERY small
    # real value probably somewhere on the order of .01

    # Because each .fast5 file is an event, group events together in
    # one fileset object that contains all files with same run_id.
    # For new run_id, new fileset, otherwise add to current fileset.

    current_run_id = (uniqueKey['tracking_id']).attrs['run_id']
    fileset = Fast5FileSet(filename, timestep, current)

    fileset.parse(Segment(current=current, start=0, end=(len(current) - 1),
                          duration=len(current), second=1000 / timestep))

    return fileset

directory = "demoFast5Reads"

maxLength = 15000
print("Showing signals below %d units in length"%maxLength)

for n,filename in enumerate(os.listdir(directory)):
    '''
    Read each fast5 file in the directory, segment, plot
    Sort by positive/neg and save the segments into a TSV if desired (see last commented section)
    metaHMM uses these TSVs to build/train/test
    '''
    print(n)

    print(filename)

    fileset = parseF5(directory + '/' + filename)


    for i in range(0,1):  #,nplots):
        filesetCopy = copy.deepcopy(fileset)
        handles = list()

        #Filter Parameters
        # order = 1
        # cutoff = 6000

        #Segmenter Parameters
        min_width = 5
        max_width = 80
        min_gain_per_sample = 0.008
        window_width = 800

        print(min_gain_per_sample)

        # print(fileset.events)
        if len(filesetCopy.events[0].current) < maxLength:  #NOTE: EDIT THIS IF YOUR EXPECTED EVENT LENGTH IS LONGER THAN 15000 units
            fig1 = plot.figure(figsize=(24, 3))
            panel = fig1.add_subplot(111)

            for event in filesetCopy.events:
                handle, = panel.plot(event.current,color="black",lw=0.2)
                handles.append(handle)

                # event.filter(order=order,cutoff=cutoff) #filter = bad
                event.parse(SpeedyStatSplit(min_width=min_width, max_width=max_width, \
                                                                min_gain_per_sample=min_gain_per_sample, \
                                                                window_width=window_width))
                # panel.plot(event.current,color="purple",lw=0.7)

            # print(fileset.events)

            sdList = list()
            meanList = list()
            prevMean = 0

            segments = None

            print("CLOSE CURRENT PLOT AND PRESS ENTER TO LOAD NEXT PLOT, OR CTRL-C TO STOP")

            for event in filesetCopy.events:
                for j,segment in enumerate(event.segments):
                    x0 = segment.start/filesetCopy.timesteps[0]*1000 -1
                    x1 = x0 + segment.duration/filesetCopy.timesteps[0]*1000
                    y0 = event.current[int(round(x0))]
                    y1 = event.current[int(round(x1))]
                    sd = segment.std
                    diff = abs(y1-y0)
                    mean = segment.mean

                    sdList.append(sd)
                    meanList.append(mean)

                    # print j,x0,x1,mean,sd,diff
                    # print(y0)

                    textColor = "red"

                    #color coding by the range of each segment to see if it indicates a bad segment
                    # if diff > 3*sd:
                    #     textColor = "red"
                    # else:
                    #     textColor = "green"

                    color = [.082, 0.282, 0.776]

                    # color coding segments by stdev to see whether stdev is an indication of bad segments
                    # if sd > 3:
                        # color = "blue"
                        # panel.text(x1, mean, "%d" % j, ha="right", va="top", color=textColor, fontsize="5")
                    # else:
                        # color = "purple"

                    # Uncomment for in-plot labelling of each segment mean
                    # panel.text(x1, mean, "%d"%mean, ha="right", va="top", color=textColor, fontsize="5")

                    # Uncomment for ugly demarcation of segment boundary:
                    # panel.plot([x0,x1],[y0,y1],marker='o',mfc="green",mew=0,markersize=3,linewidth=0)

                    #segment plotting:
                    handle, = panel.plot([x0,x1],[mean,mean],color=color,lw=0.8)
                    panel.plot([x0,x0],[prevMean,mean],color=color,lw=0.5) #  <-- uncomment for pretty square wave


                    if j==len(event.segments)-1:
                        handles.append(handle)
                        box = panel.get_position()
                        panel.set_position([box.x0, box.y0, box.width*0.95, box.height])
                        plot.legend(handles, ["Raw","Segmented"],loc='upper left', bbox_to_anchor=(1, 1))

                    prevMean = mean

                    panel.set_title("Signal")

                    panel.set_xlabel("Time (ms)")
                    panel.set_ylabel("Current (pA)")


                segments = [s for s in event.segments]

            plot.show()

            userInput = sys.stdin.readline()

            # # On-the-fly sorting of reads into true and false signals. Use this to generate a training set for HMM
            # # based on your (biased) assessment of whether the signal is an ideal translocation.
            # # AKA rapid neural net classification :)
            # # Note, this changes file names... Should be rewritten to bin into folders instead.
            #
            # print("1:SIGNAL, 2:NOISE, (NO INPUT):DO NOT SAVE")
            # if userInput == "1\n":
            #     with open("KMERS_TRUEREAD_%s.tsv"%filename[:-6], 'w') as outfile:
            #         keys = ("start", "end", "mean", "std", "min", "max", "duration")
            #         outfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%keys)
            #         for seg in segments:
            #             outfile.write('\t'.join([str(getattr(seg, k)) for k in keys])+'\n')
            #
            #         print("saving")
            #
            # elif userInput == "2\n":
            #     with open("KMERS_FALSEREAD_%s.tsv"%filename[:-6], 'w') as outfile:
            #         keys = ("start", "end", "mean", "std", "min", "max", "duration")
            #         outfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%keys)
            #         for seg in segments:
            #             outfile.write('\t'.join([str(getattr(seg, k)) for k in keys])+'\n')
            #
            #         print("saving")
            #
            # else:
            #     print("no input")
            #     pass

        else:
            print("FILE SKIPPED: too long for desired read length")
