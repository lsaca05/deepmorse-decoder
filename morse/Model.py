import tensorflow as tf
import numpy as np
import DecoderType
import sys

class Model: 
    "minimalistic TF model for Morse Decoder"

    

    def __init__(self, charList, config, decoderType=DecoderType.BestPath, mustRestore=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        global TF2
        TF2 = True
        # if TF2 is False:
        tf.compat.v1.disable_eager_execution()

        # model constants
        self.modelDir = config.value('model.name') 
        self.batchSize = config.value('model.batchSize')  # was 50 
        self.imgSize = config.value('model.imgSize')  # was (128,32)
        self.maxTextLen =  config.value('model.maxTextLen') # was 32
        self.earlyStopping = config.value('model.earlyStopping') #was 5

        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # input image batch
        if TF2 is True:
        # New keras function always omits batch size. Shape is (None, 128, 32)
            self.inputImgs = tf.keras.Input(dtype = tf.float32, shape = (self.imgSize[0], self.imgSize[1]))
        else:
            self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, self.imgSize[0], self.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        if TF2 is True:
            self.learningRate = tf.squeeze(tf.keras.Input(dtype = tf.float32, shape = [], batch_size = 1))
        else:
            self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])

        # print(self.learningRate.shape)
        # exit()
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
        # self.optimizer = tf.keras.optimizers.RMSprop(self.learningRate).minimize(self.loss, var_list = None)

        # initialize TF

        (self.sess, self.saver) = self.setupTF()

            
    def setupCNN(self):
        "create CNN layers and return output of these layers"

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        #featureVals = [1, 8, 16, 32, 32, 64]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool2d(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.cnnOut4d = pool


    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        # train new to check TF2
        if TF2 is True:
            cells = [tf.keras.layers.LSTMCell(units=numHidden) for _ in range(2)] # 2 layers
        else:
            cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers
        # stack basic cells
        if TF2 is True:
            stacked = tf.keras.layers.StackedRNNCells(cells)
        else:
            stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # bidirectional RNN
        # BxTxF -> BxTx2H
        # TODO -- Can't find equivalent line for TF 2
        ((fw, bw), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
        # 
        # ((fw,bw), _) = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(stacked))
        # no need for rnnin3d?
        # ((fw,bw), _) = tf.keras.layers.Bidirectional(stacked)
        # biStacked = tf.keras.layers.Bidirectional(stacked, merge_mode=None)
        # biStacked = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(stacked), merge_mode=None)
        # print(biStacked)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        # (None, 32, 1, 512)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        # concat = tf.expand_dims(tf.concat(biStacked, 2), 2)
        # print(((fw, bw), _))
        # print(fw)
        # print(bw)
        # print(concat)
        # exit()
        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        

    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        # SparseTensor(indices=Tensor("Placeholder:0", shape=(None, 2), dtype=int64), values=Tensor("Placeholder_1:0", shape=(None,), dtype=int32), dense_shape=Tensor("Placeholder_2:0", shape=(2,), dtype=int64))
        # TODO - This pisses off the compiler when used with eager execution. Symbolic keras cannot be used with TF SparseTensor API?
        if TF2 is True:
            self.gtTexts = tf.SparseTensor(tf.keras.Input(dtype = tf.int64, shape = [2]), tf.keras.Input(dtype = tf.int32, shape = []), tf.keras.Input(dtype=tf.int64, shape=[], batch_size=2))
        else:
            self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]) , tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        if TF2 is True:
            self.seqLen = tf.keras.Input(dtype = tf.int32, shape=[]) # Tensor("input_3:0", shape=(None,), dtype=int32)
        else:
            self.seqLen = tf.compat.v1.placeholder(tf.int32, [None]) # Tensor("Placeholder_2:0", shape=(None,), dtype=int32)

        # sequence length becomes label length and logit length, label length is none if labels is SparseTensor
        # if TF2 is True:
            # self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, logits=self.ctcIn3dTBC, label_length=None, logit_length=self.seqLen, blank_index=???))
        # else:
        self.loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        if TF2 is True:
            self.savedCtcInput = tf.keras.Input(dtype = tf.float32, shape=[None, len(self.charList) + 1], batch_size = self.maxTextLen)
        else:
            self.savedCtcInput = tf.compat.v1.placeholder(tf.float32, shape=[self.maxTextLen, None, len(self.charList) + 1])

        # if TF2 is True:
        #     self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, logits=self.savedCtcInput, label_length=None, logit_length=self.seqLen, blank_index=???)
        # else:
        self.lossPerElement = tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            if TF2 is True:
                self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50)
            else:
                self.decoder = tf.compat.v1.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            print("Loading WordBeamSearch...")
            # word_beam_search_module = tf.load_op_library('cpp/proj/TFWordBeamSearch.so')
            from word_beam_search import WordBeamSearch

            # prepare information about language (dictionary, characters in dataset, characters forming words) 
            chars = str().join(self.charList)
            wordChars = open('Volumes/Elements/' + self.modelDir+'wordCharList.txt').read().splitlines()[0]
            corpus = open('Volumes/Elements/' + self.modelDir+'corpus.txt').read()

            # TODO --  needs rewriting since imported library has changed, this or the SparseTensor/Keras TODO. Eager execution preferred.
            # decode using the "Words" mode of word beam search
            # self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, axis=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
            wbs = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
            # print(tf.nn.softmax(self.ctcIn3dTBC, axis=2))
            # exit()
            # requires eager execution to convert ctcin3dTBC to numpy array
            self.decoder = wbs.compute(tf.nn.softmax(self.ctcIn3dTBC, axis=2))


    def setupTF(self):
        "initialize TF"
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)

        sess=tf.compat.v1.Session() # TF session

        #saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        latestSnapshot = tf.train.latest_checkpoint(self.modelDir) # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + self.modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess,saver)


    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0] # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)
        #print("(indices:{}, values:{}, shape:{})".format(indices, values, shape))
        return (indices, values, shape)


    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"
        
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank=len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label==blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor 
            decoded=ctcOutput[0][0] 

            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0] # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [self.maxTextLen] * numBatchElements, self.learningRate : rate}
        #print(feedDict)
        (_, lossVal) = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal


    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"
        
        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {self.inputImgs : batch.imgs, self.seqLen : [self.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)
        
        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [self.maxTextLen] * numBatchElements}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)
        #print('inferBatch: probs:{} texts:{} '.format(probs, texts))
        return (texts, probs)
    

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, self.modelDir+'snapshot', global_step=self.snapID)
