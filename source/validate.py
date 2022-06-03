import editdistance


def validate(model, loader):
    "validate NN"
    print("Validate NN")
    loader.validationSet()
    # loader.trainSet()
    charErrorRate = float("inf")
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    wordAccuracy = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print("Batch:", iterInfo[0], "/", iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        print(recognized)

        print("Ground truth -> Recognized")
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print(
                "[OK]" if dist == 0 else "[ERR:%d]" % dist,
                '"' + batch.gtTexts[i] + '"',
                "->",
                '"' + recognized[i] + '"',
            )

    # print validation result

    try:
        charErrorRate = numCharErr / numCharTotal
        wordAccuracy = numWordOK / numWordTotal
        print(
            "Character error rate: {:4.1f}%. Word accuracy: {:4.1f}%.".format(
                charErrorRate * 100.0, wordAccuracy * 100.0
            )
        )
        print("numCharTotal:{} numWordTotal:{}".format(numCharTotal, numWordTotal))
    except:
        print("numCharTotal:{} numWordTotal:{}".format(numCharTotal, numWordTotal))
    return charErrorRate, wordAccuracy
