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

from __future__ import division, print_function

import argparse
import datetime
import os
import os.path
import uuid
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from batch import Batch
from config import Config
from decoderType import DecoderType
from filePaths import FilePaths
from generate_dataset import generate_dataset
from image import create_image
from model import Model
from morseDataset import MorseDataset
from os import listdir
from os.path import isfile, join

# Read WAV file containing Morse code and create 256x1 (or 16x16) tiles (256 samples/4 seconds)
from train import train
from validate import validate


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


# import sounddevice as sd


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return an open file handle


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = create_image(fnImg, model.imgSize)
    plt.imshow(img, cmap=cm.Greys_r)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print("Recognized:", '"' + recognized[0] + '"')
    print("Probability:", probability[0])
    print(recognized)


def infer_image(model, o):
    im = o[0::1].reshape(1, 256)
    im = im * 256.0
    img = cv2.resize(im, model.imgSize, interpolation=cv2.INTER_AREA)
    fname = f"dummy{uuid.uuid4().hex}.png"
    cv2.imwrite(fname, img)
    img = cv2.transpose(img)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    return fname, recognized, probability


# from pyAudioAnalysis.audioSegmentation import silence_removal
# deprecated, part of --experiment
# def infer_file(model, fname):
#     print(f"SILENCE REMOVAL:{remove_silence}")
#     if remove_silence:
#         print()
#         tone = find_peak(fname)
#         [Fs, x] = wavfile.read(fname)
#         segments = silence_removal(x, Fs, 0.25, 0.05, 0.2, 0.2, False)
#         for start, stop in segments:
#             print("*" * 80, f"start:{start}, stop:{stop} dur:{stop-start}")
#             o, dur = process_audio_file2(fname, start, stop, tone)
#             start_time = datetime.datetime.now()
#             iname, recognized, probability = infer_image(model, o[0:256])
#             stop_time = datetime.datetime.now()
#             if True:  # probability[0] > 0.00005:
#                 print(f"File:{iname}")
#                 print("Recognized:", '"' + recognized[0] + '"')
#                 print("Probability:", probability[0])
#                 print("Duration:{}".format(stop_time - start_time))
#         return
#     else:
#         sample = 4.0
#         start = 0.0
#         tone = find_peak(fname)
#         o, dur = process_audio_file(fname, start, sample, tone)
#         while start < (dur - sample):
#             print(start, dur)
#             im = o[0::1].reshape(1, 256)
#             im = im * 256.0
#             img = cv2.resize(im, model.imgSize, interpolation=cv2.INTER_AREA)
#             cv2.imwrite(f"dummy{start}.png", img)

#             img = cv2.transpose(img)

#             batch = Batch(None, [img])
#             start_time = datetime.datetime.now()
#             (recognized, probability) = model.inferBatch(batch, True)
#             stop_time = datetime.datetime.now()
#             if probability[0] > 0.0000:
#                 print("Recognized:", '"' + recognized[0] + '"')
#                 print("Probability:", probability[0])
#                 print("Duration:{}".format(stop_time - start_time))
#             start += 1.0 / 1
#             o, dur = process_audio_file(fname, start, sample, tone)


def main():
    "main function"

    global remove_silence
    remove_silence = False
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    # parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    # parser.add_argument(
    # "--wordbeamsearch",
    # help="use word beam search instead of best path decoding",
    # action="store_true"
    # )
    parser.add_argument(
        "--generate",
        help="generate a Morse dataset of random words",
        action="store_true",
    )
    parser.add_argument(
        "--experiments",
        help="generate a set of experiments using config files",
        action="store_true",
    )
    parser.add_argument("--silence", help="remove silence", action="store_true")
    parser.add_argument(
        "-f",
        dest="filename",
        required=False,
        help="input audio file ",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    args = parser.parse_args()

    # read configs for current training/validation/inference job
    config = Config("model_arrl7.yaml")

    decoderType = DecoderType.WordBeamSearch
    # decoderType = DecoderType.BeamSearch
    # decoderType = DecoderType.BestPath

    # if args.beamsearch:
    #    decoderType = DecoderType.BeamSearch
    # elif args.wordbeamsearch:
    #    decoderType = DecoderType.WordBeamSearch
    if args.experiments:
        print("*" * 80)
        print(f"Looking for model files in {config.value('experiment.fnExperiments')}")
        experiments = [
            f
            for f in listdir(config.value("experiment.fnExperiments"))
            if isfile(join(config.value("experiment.fnExperiments"), f))
        ]
        print(experiments)
        for filename in experiments:
            tf.reset_default_graph()
            config = Config(FilePaths.fnExperiments + filename)
            generate_dataset(config)
            decoderType = DecoderType.WordBeamSearch
            loader = MorseDataset(config)
            model = Model(loader.charList, config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            with open(FilePaths.fnResults, "a+") as fr:
                fr.write(
                    "\nexperiment:{} loss:{} charErrorRate:{} wordAccuracy:{}".format(
                        filename, min(loss), min(charErrorRate), max(wordAccuracy)
                    )
                )
            tf.reset_default_graph()
        return
    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        # loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        # loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        decoderType = DecoderType.WordBeamSearch
        decoderType = DecoderType.BeamSearch
        loader = MorseDataset(config)

        # save characters of model for inference mode
        open(config.value("experiment.fnCharList"), "w").write(
            str().join(loader.charList)
        )

        # save words contained in dataset into file
        open(config.value("experiment.fnCorpus"), "w").write(
            str(" ").join(loader.trainWords + loader.validationWords)
        )

        # execute training or validation
        if args.train:
            model = Model(loader.charList, config, decoderType)
            loss, charErrorRate, wordAccuracy = train(model, loader)
            plt.figure(figsize=(20, 10))
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
            print("SILENCE REMOVAL ON")
            remove_silence = True
        config = Config("model_arrl7.yaml")
        print("*" * 80)
        print(open(config.value("experiment.fnAccuracy")).read())
        start_time = datetime.datetime.now()
        model = Model(
            open(config.value("experiment.fnCharList")).read(),
            config,
            decoderType,
            mustRestore=True,
        )
        print("Loading model took:{}".format(datetime.datetime.now() - start_time))
        # deprecated
        # infer_file(model, args.filename)


if __name__ == "__main__":
    main()
