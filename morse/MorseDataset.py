import os
import random

from batch import Batch
from image import create_image
from sample import Sample


class MorseDataset:
    def __init__(self, config):
        "loader for dataset at given location, preprocess images and text according to parameters"
        # filePath, batchSize, imgSize, maxTextLen
        self.filePath = config.value("model.directory")
        # assert self.filePath[-1]=='/'
        self.batchSize = config.value("model.batchSize")
        self.imgSize = config.value("model.imgSize")
        self.maxTextLen = config.value("model.maxTextLen")
        self.samples = []
        self.dataAugmentation = False
        self.currIdx = 0

        try:
            os.makedirs(self.filePath)
        except OSError:
            print("Error: cannot create ", self.filePath)
            # if not os.path.isdir(filePath):
            #    raise
        print(f"MorseDataset: loading {config.value('morse.fnTrain')}")
        f = open(config.value("morse.fnTrain"), "r")
        chars = set()
        bad_samples = []

        # read all lines in the file
        for line in f:
            # ignore comment line
            if not line or line[0] == "#":
                continue
            # print(line)
            # lineSplit = line.strip().split(' ')
            # ensure that words starting with a space are correctly split, exclude last character (endline)
            # TODO - figure out workaround for "/" and "?".
            lineSplit = line.strip("\n").split(".wav ")
            # print(lineSplit)
            assert len(lineSplit) >= 2, "line is {}".format(line)

            # filenames: audio/*.wav
            # fileNameAudio = lineSplit[0]
            fileNameAudio = lineSplit[0] + ".wav"
            # Ground Truth text - open files and append to samples
            #

            gtText = self.truncateLabel(" ".join(lineSplit[1:]), self.maxTextLen)
            gtText = gtText + " "
            print(gtText)
            chars = chars.union(set(list(gtText)))

            # put sample into list
            # print("sample text length:{} {}".format(len(gtText), gtText))
            self.samples.append(Sample(gtText, fileNameAudio))

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 25000

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))
        file_name = config.value("experiment.fnCharList")
        print(f"file:{file_name}")
        open(file_name, "w").write(str().join(self.charList))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = False  # was True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[: self.numTrainSamplesPerEpoch]

    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [
            create_image(self.samples[i].filePath, self.imgSize, self.dataAugmentation)
            for i in batchRange
        ]
        # imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)
