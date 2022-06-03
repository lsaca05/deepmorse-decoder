import sys

from numpy import Inf, NaN, arange, array, asarray, isscalar
from scipy.io import wavfile
from scipy.signal import periodogram


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit("Input vectors v and x must have same length")

    if not isscalar(delta):
        sys.exit("Input argument delta must be a scalar")

    if delta <= 0:
        sys.exit("Input argument delta must be positive")

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def find_peak(fname):
    """Find the signal frequency and maximum value"""
    # print("find_peak",fname)
    Fs, x = wavfile.read(fname)
    f, s = periodogram(x, Fs, "blackman", 8192, "linear", False, scaling="spectrum")
    threshold = max(s) * 0.8  # only 0.4 ... 1.0 of max value freq peaks included
    maxtab, mintab = peakdet(
        abs(s[0 : int(len(s) / 2 - 1)]), threshold, f[0 : int(len(f) / 2 - 1)]
    )
    try:
        val = maxtab[0, 0]
    # specify exception if possible
    except:
        print("Error: {}".format(maxtab))
        val = 600.0
    return val
