# 24487 words in alphabetical order
# https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain
#

import random
import re
import uuid

from source.morse import Morse


def generate_dataset(config):
    "generate audio dataset from locally generated text file"
    fnTrain = config.value("morse.fnTrain")
    fnAudio = config.value("morse.fnAudio")
    code_speed = config.value("morse.code_speed")
    SNR_DB = config.value("morse.SNR_dB")
    count = config.value("morse.count")
    length_seconds = config.value("morse.length_seconds")
    word_max_length = config.value("morse.word_max_length")
    words_in_sample = config.value("morse.words_in_sample")
    print("SNR_DB:{}".format(SNR_DB))
    file = open("morse/samples_weighted.txt").read()

    with open(fnTrain, "w") as mf:
        # print(rv)
        # words = rv.text.split("\n")
        # words = rv.split("\n")
        words = file.split("\n")
        # wordcount = len(words)
        words = [w.upper() for w in words if len(w) <= word_max_length]
        for i in range(count):  # count of samples to generate
            sample = random.sample(words, words_in_sample)
            line = " ".join(sample)
            phrase = re.sub(r"[\'.&]", "", line)
            # if len(phrase) <= 1:
            if len(phrase) < 1:
                continue
            speed = random.sample(code_speed, 1)
            SNR = random.sample(SNR_DB, 1)
            audio_file = "{}SNR{}WPM{}-{}-{}.wav".format(
                fnAudio, SNR[0], speed[0], phrase, uuid.uuid4().hex
            )
            # print(Morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, 8, False))
            # morse(phrase, audio_file, SNR[0], 600, 8000, speed[0], length_seconds, 8, False)
            # mf.write(audio_file+' '+phrase+'\n')
            if phrase.isspace():
                print("Sample was all spaces, didn't write")
            elif not Morse(
                phrase,
                audio_file,
                SNR[0],
                600,
                8000,
                speed[0],
                length_seconds,
                4,
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
