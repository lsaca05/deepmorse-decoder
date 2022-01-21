
# coding: utf-8

"""
A Morse Decoder implementation using TensorFlow library.
Learn to classify Morse code sequences using a neural network with CNN + LSTM + CTC

Adapted by:  Mauri Niininen (AG1LE) for Morse code learning
Improved by: Lucas Saca, as part of a Walla Walla University Edward F. Cross School of Engineering Capstone Project.

From: Handwritten Text Recognition (HTR) system implemented with TensorFlow.
by Harald Scheidl
See: https://github.com/githubharald/SimpleHTR
"""

from __future__ import division
from __future__ import print_function

import sys
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import random
from numpy.random import normal
import numpy as np
#from morse import Morse
# import yaml                       # Config
# from functools import reduce      # Config
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime


    
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
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
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
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
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


# Read WAV file containing Morse code and create 256x1 (or 16x16) tiles (256 samples/4 seconds)
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import numpy as np

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, periodogram
#from peakdetect import peakdet  # download peakdetect from # https://gist.github.com/endolith/250860

def find_peak(fname):
    """Find the signal frequency and maximum value"""
    #print("find_peak",fname)
    Fs, x = wavfile.read(fname)
    f,s = periodogram(x, Fs,'blackman',8192,'linear', False, scaling='spectrum')
    threshold = max(s)*0.8  # only 0.4 ... 1.0 of max value freq peaks included
    maxtab, mintab = peakdet(abs(s[0:int(len(s)/2-1)]), threshold,f[0:int(len(f)/2-1)] )
    try:
        val = maxtab[0,0]
    except:
        print("Error: {}".format(maxtab))
        val = 600.
    return val

# Fs should be 8000 Hz 
# with decimation down to 125 Hz we get 8 msec / sample
# with WPM equals to 20 => Tdit = 1200/WPM = 60 msec   (time of 'dit')
# 4 seconds equals 256 samples ~ 66.67 Tdits 
# word 'PARIS' is 50 Tdits

def demodulate(x, Fs, freq):
    """return decimated and demodulated audio signal envelope at a known CW frequency """
    t = np.arange(len(x))/ float(Fs)
    mixed =  x*((1 + np.sin(2*np.pi*freq*t))/2 )

    #calculate envelope and low pass filter this demodulated signal
    #filter bandwidth impacts decoding accuracy significantly 
    #for high SNR signals 40 Hz is better, for low SNR 20Hz is better
    # 25Hz is a compromise - could this be made an adaptive value?
    low_cutoff = 25. # 25 Hz cut-off for lowpass
    wn = low_cutoff/ (Fs/2.)    
    b, a = butter(3, wn)  # 3rd order butterworth filter
    z = filtfilt(b, a, abs(mixed))
    
    decimate = int(Fs/64) # 8000 Hz / 64 = 125 Hz => 8 msec / sample 
    Ts = 1000.*decimate/float(Fs)
    o = z[0::decimate]/max(z)
    return o

def process_audio_file(fname,x,y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal)/Fs
    o = demodulate(signal[int(Fs*(x)):int(Fs*(x+y))], Fs, tone)
    #print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur

def process_audio_file2(fname,x,y, tone):
    """return demodulated clip from audiofile from x to y seconds at tone frequency,  as well as duration of audio file in seconds"""
    Fs, signal = wavfile.read(fname)
    dur = len(signal)/Fs
    if y - x < 4.0:
        end = x + 4.0
        xi = int(Fs*x)
        yi = int(Fs*y)
        ei = int(Fs*end)
        pad = np.zeros(ei-yi)
        print(f"dur:{dur}x:{x},y:{y}, end:{end}, xi:{xi}, yi:{yi}, ei:{ei}")
        signal = np.insert(signal, slice(yi, ei), pad)
        y = end
        
    o = demodulate(signal[int(Fs*(x)):int(Fs*(x+y))], Fs, tone)
    #print("Fs:{} total duration:{} sec start at:{} seconds, get first {} seconds".format(Fs, dur,x,y))
    return o, dur

# Read morse.wav from start_time=0 duration=4 seconds
# save demodulated/decimated signal (1,256) to morse.npy 
# options:
# decimate: Fs/16   Fs/64  Fs/64
# duration: 2        4       16
# imgsize : 32       256    1024

"""
filename = "audio/2c1018174f794091916353937fc9f518.wav"
tone = find_peak(filename)
print("tone:{}".format(tone))

o,dur = process_audio_file(filename,0,4, tone)
np.save("morse.npy", o, allow_pickle=False)
im = o[0::1].reshape(1,256)
#o[10:32] = 0.
#im = o[0::1].reshape(1,32)

plt.figure(figsize=(20,10))
plt.subplot(2, 1, 1)
plt.plot(o[0::1])
#plt.annotate('N',xy=(25, 1))
#plt.annotate('O',xy=(90, 1))
#plt.annotate('W',xy=(150, 1))
#plt.annotate('2',xy=(255, 1))
plt.ylabel('amplitude')
plt.xlabel('time')
plt.subplot(2, 1, 2)
plt.imshow(im,cmap = cm.Greys_r)
plt.xlabel('time')
plt.show()
"""


import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import write
#import sounddevice as sd
import matplotlib.pyplot as plt 


def create_image2(filename, imgSize, dataAugmentation=False):

    RATE = 8000
    FORMAT = pyaudio.paInt16  # conversion format for PyAudio stream
    CHANNELS = 1  # microphone audio channels
    CHUNK_SIZE = 8192  # number of samples to take per read
    SAMPLE_LENGTH = int(CHUNK_SIZE * 1000 / RATE)  # length of each sample in ms
    SAMPLES_PER_FRAME = 4 

    rate, data = wavfile.read(fname)
    arr2D, freqs, bins = get_specgram(data, rate)

    fig, ax = plt.subplots(1,1)
    extent = (bins[0], bins[-1] * SAMPLES_PER_FRAME, freqs[-1], freqs[0])
    im = ax.imshow(
        arr2D,
        aspect="auto",
        extent=extent,
        interpolation="none",
        cmap="Greys",
        norm=None,
    )
    plt.show()



    im_data = im.get_array()
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)

    # Get the image data array shape (Freq bins, Time Steps)
    shape = im_data.shape

    # Find the CW spectrum peak - look across all time steps
    f = int(np.argmax(im_data[:]) / shape[1])

    # Create a 32x128 array centered to spectrum peak
    img = cv2.resize(im_data[f - 16 : f + 16][0:128], (128, 32))
    if img.shape == (32, 128):
        cv2.imwrite("dummy.png", img)

   # transpose for TF
    img = cv2.transpose(img)


    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img

    return  img


def create_image(filename, imgSize, dataAugmentation=False):
    
    # get image name from audio file name - assumes 'audio/filename.wav' format
    #name = filename.split('/')
    imgname = filename+".png"
    
    # Load  image in grayscale if exists
    img = cv2.imread(imgname,0)
        
    if img is None:
        #print('.') #could not load image:{} processing audio file'.format(imgname))

        # find the Morse code peak tone 
        tone = find_peak(filename)
        # sample = 16 seconds from audio file into output => (1,1024) 
        # sample = 4 seconds from audio file into output => (1,256) 
        sample = 4 
        o,dur = process_audio_file(filename,0,sample, tone)
        # reshape output into image and resize to match the imgSize of the model (128,32)
        #im = o[0::1].reshape(4,256)
        im = o[0::1].reshape(1,256)
        im = im*256.
        img = cv2.resize(im, imgSize, interpolation = cv2.INTER_AREA)
        # save to file 
        retval = cv2.imwrite(imgname,img)
        if not retval:
            # going to see a lot of this -- tries to make image from existing image.
            print('Error in writing image:{} retval:{}'.format(imgname,retval))

    
    """
    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5) # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    
    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.zeros([ht, wt]) #* 255
    target[0:newSize[1], 0:newSize[0]] = img
    """
        
    # transpose for TF
    img = cv2.transpose(img)


    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    

    
    # transpose to match tensorflow requirements
    return img




import sys
import argparse
import cv2
import editdistance
import os.path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return an open file handle

def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = create_image(fnImg, model.imgSize)
    plt.imshow(img,cmap = cm.Greys_r)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    print(recognized)


#from pyAudioAnalysis.audioSegmentation import silence_removal

def infer_image(model, o):
    im = o[0::1].reshape(1,256)
    im = im*256.
    img = cv2.resize(im, model.imgSize, interpolation = cv2.INTER_AREA)
    fname =f'dummy{uuid.uuid4().hex}.png'
    cv2.imwrite(fname,img)
    img = cv2.transpose(img)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    return fname, recognized, probability


   
def infer_file(model, fname):
    print(f"SILENCE REMOVAL:{remove_silence}")
    if remove_silence:
        print()
        tone = find_peak(fname)
        [Fs,x] = wavfile.read(fname)
        segments = silence_removal(x, Fs, 0.25, 0.05, 0.2, 0.2, False)
        for start, stop in segments: 
            print("*"*80,f"start:{start}, stop:{stop} dur:{stop-start}")
            o,dur = process_audio_file2(fname, start, stop, tone)
            start_time = datetime.datetime.now()
            iname, recognized, probability = infer_image(model, o[0:256])
            stop_time = datetime.datetime.now()
            if True: #probability[0] > 0.00005:
                print(f'File:{iname}')
                print('Recognized:', '"' + recognized[0] + '"')
                print('Probability:', probability[0])
                print('Duration:{}'.format(stop_time-start_time))
        return 
    else:        
        sample = 4.0 
        start = 0.
        tone = find_peak(fname)
        o,dur = process_audio_file(fname,start,sample, tone)
        while start < (dur - sample):
            print(start,dur)
            im = o[0::1].reshape(1,256)
            im = im*256.
            img = cv2.resize(im, model.imgSize, interpolation = cv2.INTER_AREA)
            cv2.imwrite(f'dummy{start}.png',img)

            img = cv2.transpose(img)

            batch = Batch(None, [img])
            start_time = datetime.datetime.now()
            (recognized, probability) = model.inferBatch(batch, True)
            stop_time = datetime.datetime.now()
            if probability[0] > 0.0000:
                print('Recognized:', '"' + recognized[0] + '"')
                print('Probability:', probability[0])
                print('Duration:{}'.format(stop_time-start_time))
            start += 1./1
            o,dur = process_audio_file(fname,start,sample, tone)

 
import Config
import DecoderType
import FilePaths
import generate_dataset
import MorseDataset
import Model
import train
import validate

def main():
    "main function"

    global remove_silence
    remove_silence = False
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    #parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    #parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    parser.add_argument("--generate", help="generate a Morse dataset of random words", action="store_true")
    parser.add_argument("--experiments", help="generate a set of experiments using config files", action="store_true")
    parser.add_argument("--silence", help="remove silence", action="store_true")
    parser.add_argument("-f", dest="filename", required=False,
                    help="input audio file ", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))

    args = parser.parse_args()

    config = Config('model_arrl4.yaml') #read configs for current training/validation/inference job

    decoderType = DecoderType.WordBeamSearch
    #decoderType = DecoderType.BeamSearch
    #decoderType = DecoderType.BestPath
    
    #if args.beamsearch:
    #    decoderType = DecoderType.BeamSearch
    #elif args.wordbeamsearch:
    #    decoderType = DecoderType.WordBeamSearch
    if args.experiments:
        print("*"*80)
        print(f"Looking for model files in {config.value('experiment.fnExperiments')}")
        experiments = [f for f in listdir(config.value("experiment.fnExperiments")) if isfile(join(config.value("experiment.fnExperiments"), f))]
        print(experiments)
        for filename in experiments:
            tf.reset_default_graph()
            config = Config(FilePaths.fnExperiments+filename)
            generate_dataset(config)
            decoderType = DecoderType.WordBeamSearch
            loader = MorseDataset(config)
            model = Model(loader.charList, config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            with open(FilePaths.fnResults, 'a+') as fr:
                fr.write("\nexperiment:{} loss:{} charErrorRate:{} wordAccuracy:{}".format(filename, min(loss), min(charErrorRate), max(wordAccuracy)))
            tf.reset_default_graph()
        return
    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        #loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        decoderType = DecoderType.WordBeamSearch
        decoderType = DecoderType.BeamSearch
        loader = MorseDataset(config)

        # save characters of model for inference mode
        open(config.value("experiment.fnCharList"), 'w').write(str().join(loader.charList))
                
        # save words contained in dataset into file
        open(config.value("experiment.fnCorpus"), 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
        
        # execute training or validation
        if args.train:
            model = Model(loader.charList, config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            plt.figure(figsize=(20,10))
            plt.subplot(3, 1, 1)
            plt.title("Character Error Rate")
            plt.plot(charErrorRate)
            plt.subplot(3, 1, 2)
            plt.title("Word Accuracy")
            plt.plot(wordAccuracy)
            plt.subplot(3, 1, 3)
            plt.title("Loss")
            plt.plot(loss)
            plt.show()
        elif args.validate:
            model = Model(loader.charList, config, decoderType, mustRestore=True)
            validate(model, loader)
    elif args.generate:
        generate_dataset(config)

    # infer text on test audio file
    else:
        if args.silence:
            print(f"SILENCE REMOVAL ON")
            remove_silence = True
        config = Config('model_arrl4.yaml')
        print("*"*80)
        print(open(config.value("experiment.fnAccuracy")).read())
        start_time = datetime.datetime.now()
        model = Model(open(config.value("experiment.fnCharList")).read(), config, decoderType, mustRestore=True)
        print("Loading model took:{}".format(datetime.datetime.now()-start_time))
        infer_file(model, args.filename)


if __name__ == "__main__":
    main()

