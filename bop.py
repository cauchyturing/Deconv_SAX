#!/usr/bin/python


import math
from normal_cutoff import normal_cutoffs

#
# This is an implementation of the bag-of-patterns representation for
# time series as described in the paper "Finding Structural Similarity 
# in Time Series Data Using Bag-of-Patterns Representation" by Jessica
# Lin and Yuan Li.  At its core is the SAX discretization method described
# in numerous papers by Jessica Lin and Eamonn Keogh.  The canonical 
# reference is "Experiencing SAX: a Novel Symbolic Representation of Time 
# Series", by Lin, Keogh, Wei, and Lonardi.
#
# All time series are represented as lists of numbers.
#
# SAX words are represented as lists of symbols so that they can be compared
# using the MinDist metric described in the Li, Keogh, Wei, and Lombardi 
# paper.  Those words are represented as strings when they are stored in bags
# of patterns because similarity between bags is based on word frequencies
# rather than the words themselves.
#



#
# Return a list of all subsequences of a time series obtained by a sliding 
# window.
#
#   series - The time series (list of values)
#   n - The size of the sliding window
#   step - If step is greater than or equal to 1, the window is advanced
#          by that many observations as it slides along the series.  If 
#          step is less than 1, it is interpreted as the fraction of n by
#          which the window should be advanced.
#
# The return value is a list of lists with each sublist containing a window of
# data.  The sublists appear in the order in which they were extracted by 
# sliding the window left-to-right.
#
def window_time_series(series, n, step = 1):
#    print "in window_time_series",series
    if step < 1.0:
        step = max(int(step * n), 1)
    return [series[i:i+n] for i in range(0, len(series) - n + 1, step)]



#
# Return a copy of a time series in which the values have been standardized
# to have mean 0 and standard deviation 1
#
#   series - The time series
#
def standardize(series):

    # Compute mean and standard deviation
    n = float(len(series))
    mean = sum(series) / n
    variance = sum([(value - mean)**2 for value in series]) / (n - 1)
    stddev = math.sqrt(variance)

    # Just in case the data have no variation
    if stddev == 0:
        stddev = 1.0

    # Normalize via z transform
    return [(value - mean)/stddev for value in series]



#
# Create a piecewise aggregate approximation of a time series by computing
# the means of the values in adjacent, non-overlapping windows.
#
#   series - The time series to discretize
#   now - Number of windows (see below)
#   opw - Observations per window (see below)
#
# Eaxctly one of now or opw must be None, with the other being an integer.  
# If now is specified, then the original time series will be converted into 
# a sequence of now values.  If opw is specified, then the original time 
# series will be converted into a sequence of values such that each value is 
# the mean of opw observations in the time series.
#
def paa(series, now, opw):
    if now == None:
        now = len(series) / opw
    if opw == None:
        opw = len(series) / now
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]



#
# normal_cutoffs[i] is a list of i+1 cut points for a standard normal 
# distribution such that cutting the distribution at those points yields
# i+2 sections that are mutually exclusive, exhaustive, and equiprobable.




#
# Discretize a time series using the SAX method.  The input series is 
# converted into a list of symbols chosen from an alphabet such that the
# number of occurrences of each symbol is roughly equiprobable.
#
#   series - The time series to discretize
#   nos - Number of symbols (see below)
#   ops - Observations per symbol (see below)
#   a - The alphabet size
#   nostd - If true then do not standardize the data
#
# Eaxctly one of nos or ops must be None, with the other being an integer.  
# If nos is specified, then the original time series will be converted into 
# a sequence of nos symbols.  If ops is specified, then the original time 
# series will be converted into a sequence of symbols such that each symbol 
# spans ops observations in the time series.
#
# The nostd flag makes it possible to run SAX on a subsequence of a larger
# time series that has already been standardized.  Do not run SAX on time 
# series that are not standardized with nostd = True.
#
# The return value is a SAX word represented as a list of symbols.
#
def sax(series, nos, ops, a, nostd = False):

    if not nostd:
        series = standardize(series)
#    print "in sax: series",series
#    print "SERIES"
#    print series
    means = paa(series, nos, ops)
#    print "PAA"
#    print means

    # Convert numeric values to integers between 0 and a-1 (i.e., the symbols
    # of the alphabet)
    cutoffs = normal_cutoffs[a - 2]
    retval = []
    for mean in means:
        i = 0
        for cutoff in cutoffs:
            if mean > cutoff:
                i = i + 1
        retval.append(i)
    
#    print "in sax: sax",retval
#    print "SAX"
#    print retval
    return retval



#
# Compute the MinDist distance metric between two SAX words represented as
# lists of symbols.  
#
#   sax_word_1 - The first SAX word
#   sax_word_2 - The second SAX word
#   a - The size of the SAX alphabet used when creating the words
#
def min_dist(sax_word_1, sax_word_2, a):

    dist = 0.0

    for i in range(len(sax_word_1)):
        r = sax_word_1[i]
        c = sax_word_2[i]
        if abs(r - c) > 1:
            dist = dist + normal_cutoffs[a - 2][max(r, c) - 1]
            dist = dist - normal_cutoffs[a - 2][min(r, c)]

    return dist



#
# Turn the values in a list into a string where each value is delimited
# by some other string
#
#   l - The list
#   delim - The delimiter
#
def list_to_delimted_string(l, delim):
    s = ''
    for item in l:
        s = s + str(item) + delim
    return s[:-len(delim)]



#
# Convert a time series into a list of SAX words by sliding a window over
# the series and running SAX on each window, yielding one SAX word per window
#
#   series - The series to convert
#   n - The length of the sliding window
#   step - See window_time_series for an explanation of this parameter
#   nos, ops, a, nostd - Parameters to SAX
#
def sax_words(series, n, step, nos, ops, a, nostd = False):
#    print "in sax_words", series

    windows = window_time_series(series, n, step)
    windows = [sax(window, nos, ops, a, nostd) for window in windows]

    return [list_to_delimted_string(window, '_') for window in windows]

    

#
# Given a list of SAX words, reduce runs of the same word to a single
# occurrence
#
#   words - The list of SAX words
#
def numerosity_reduction(words):
    reduced = [words[0]]

    for word in words:
        if word <> reduced[-1]:
            reduced.append(word)

    return reduced



#
# Create a bag-of-patterns from a list of SAX words, which is a dictionary
# keyed by word with values equal to the number of times each word occurs
#
#   sax_words - List of SAX words
#
def bop(sax_words):
    dict = {}

    # Count occurrences of words
    for word in sax_words:
        if not word in dict:
            dict[word] = 0
        dict[word] = dict[word] + 1
        
    return dict

