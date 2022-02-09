# 24487 words in alphabetical order
# https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain
#

# import os
import random
import re
import uuid

# import requests

from morse import morse


def generate_dataset(config):
    "generate audio dataset from a dictionary of random words"
    # URL = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
    # # Header needed as part of request, website blocks script/non-browser requests it seems
    # hdr = {'User-Agent' : 'Mozilla/5.0'}
    filePath = config.value("model.name")
    fnTrain = config.value("morse.fnTrain")
    fnAudio = config.value("morse.fnAudio")
    code_speed = config.value("morse.code_speed")
    SNR_DB = config.value("morse.SNR_dB")
    count = config.value("morse.count")
    length_seconds = config.value("morse.length_seconds")
    word_max_length = config.value("morse.word_max_length")
    words_in_sample = config.value("morse.words_in_sample")
    print("SNR_DB:{}".format(SNR_DB))
    # rv = requests.get(URL, headers=hdr)
    rv = open("morse/new_samples.txt").read()
    # if rv.status_code == 200:
    #     try:
    #         os.makedirs(filePath)
    #     except OSError:
    #         print("Error: cannot create ", filePath)

    with open(fnTrain, "w") as mf:
        # print(rv)
        # words = rv.text.split("\n")
        words = rv.split("\n")
        wordcount = len(words)
        words = [w.upper() for w in words if len(w) <= word_max_length]
        for i in range(count):  # count of samples to generate
            sample = random.sample(words, words_in_sample)
            line = " ".join(sample)
            phrase = re.sub(r"[\'.&]", "", line)
            if len(phrase) <= 1:
                continue
            speed = random.sample(code_speed, 1)
            SNR = random.sample(SNR_DB, 1)
            audio_file = "{}SNR{}WPM{}-{}-{}.wav".format(
                fnAudio, SNR[0], speed[0], phrase, uuid.uuid4().hex
            )
            # print(morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, 8, False))
            # morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, 8, False)
            # mf.write(audio_file+' '+phrase+'\n')
            if not morse(
                phrase,
                audio_file,
                SNR[0],
                600,
                8000,
                speed[0],
                length_seconds,
                8,
                False,
            ).all():
                print(audio_file, phrase + str(" TOO LONG, didn't write"))
            else:
                mf.write(audio_file + " " + phrase + "\n")
            # print(audio_file, phrase)
        print("completed {} files".format(count))
    # elif rv.status_code == 404:
    #     print("ERROR: URL Page Not Found")
    #     print("Random morse word generation failed.")
    # else:
    #     print("ERROR: Request status code " + str(rv.status_code))
    #     print("Random morse word generation failed.")
