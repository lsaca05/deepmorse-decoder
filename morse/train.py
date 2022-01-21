import datetime
import validate
import FilePaths

def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best validation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = model.earlyStopping  # stop training after this number of epochs without improvement
    accLoss = []
    accChrErrRate = []
    accWordAccuracy = []
    start_time = datetime.datetime.now()
    while True:
        epoch += 1
        print('Epoch: {} Duration:{}'.format(epoch, datetime.datetime.now()-start_time))

        # train
        print('Train NN - imgSize',model.imgSize)
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
            accLoss.append(loss)

        # validate
        charErrorRate, wordAccuracy = validate(model, loader)
        accChrErrRate.append(charErrorRate)
        accWordAccuracy.append(wordAccuracy)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate {:4.1f}% improved, save model'.format(charErrorRate*100.))
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: {:4.1f}% word accuracy: {:4.1f}'.format(charErrorRate*100.0, wordAccuracy*100.))
        else:
            noImprovementSince += 1
            print('Character error rate {:4.1f}% not improved in last {} epochs'.format(charErrorRate*100., noImprovementSince))

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since {} epochs. Training stopped.'.format(earlyStopping))
            break
    end_time = datetime.datetime.now()
    print("Total training time was {}".format(end_time-start_time))
    return accLoss, accChrErrRate, accWordAccuracy
