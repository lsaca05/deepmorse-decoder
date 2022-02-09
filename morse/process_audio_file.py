import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Fs should be 8000 Hz
# with decimation down to 125 Hz we get 8 msec / sample
# with WPM equals to 20 => Tdit = 1200/WPM = 60 msec   (time of 'dit')
# 4 seconds equals 256 samples ~ 66.67 Tdits
# word 'PARIS' is 50 Tdits


def demodulate(x, Fs, freq):
    """return decimated and demodulated audio signal envelope at a known CW frequency"""
    t = np.arange(len(x)) / float(Fs)
    mixed = x * ((1 + np.sin(2 * np.pi * freq * t)) / 2)

    # calculate envelope and low pass filter this demodulated signal
    # filter bandwidth impacts decoding accuracy significantly
    # for high SNR signals 40 Hz is better, for low SNR 20Hz is better
    # 25Hz is a compromise - could this be made an adaptive value?
    low_cutoff = 25.0  # 25 Hz cut-off for lowpass
    wn = low_cutoff / (Fs / 2.0)
    b, a = butter(3, wn)  # 3rd order butterworth filter
    z = filtfilt(b, a, abs(mixed))

    decimate = int(Fs / 64)  # 8000 Hz / 64 = 125 Hz => 8 msec / sample
    Ts = 1000.0 * decimate / float(Fs)
    o = z[0::decimate] / max(z)
    return o


def process_audio_file(fname, x, y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal) / Fs
    o = demodulate(signal[int(Fs * (x)) : int(Fs * (x + y))], Fs, tone)
    # print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur


def process_audio_file2(fname, x, y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal) / Fs
    if y - x < 4.0:
        end = x + 4.0
        xi = int(Fs * x)
        yi = int(Fs * y)
        ei = int(Fs * end)
        pad = np.zeros(ei - yi)
        print(f"dur:{dur}x:{x},y:{y}, end:{end}, xi:{xi}, yi:{yi}, ei:{ei}")
        signal = np.insert(signal, slice(yi, ei), pad)
        y = end

    o = demodulate(signal[int(Fs * (x)) : int(Fs * (x + y))], Fs, tone)
    # print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur
