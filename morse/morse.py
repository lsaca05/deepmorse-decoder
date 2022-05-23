import random

import numpy as np
from morseCode import MorseCode
from scipy.io.wavfile import write
# import sounddevice as sd


def morse(
    text,
    file_name=None,
    SNR_dB=20,
    f_code=600,
    Fs=8000,
    code_speed=20,
    length_seconds=4,
    total_seconds=8,
    play_sound=True,
):
    """
    # MORSE converts text to playable morse code in wav format
    #
    # SYNTAX
    # morse(text)
    # morse(text,file_name),
    # morse(text,file_name,SNR_dB),
    # morse(text, file_name,SNR_dB,code_frequency),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate, code_speed_wpm, zero_fill_to_N),
    # morse(text, file_name,SNR_dB,code_frequency,sample_rate, code_speed_wpm, zero_fill_to_N, play_sound),
    #
    # Description:
    #
    #   If the wave file name is specified, then the function will output a wav
    #   file with that file name.  If only text is specified, then the function
    #   will only play the morse code wav file without saving it to a wav file.
    #   If a snr is specified, zero mean addative white Gaussian
    #   noise is added
    #
    # Examples:
    #
    #   morse('Hello'),
    #   morse('How are you doing my friend?','morsecode.wav'),
    #   morse('How are you doing my friend?','morsecode.wav', 20),
    #   morse('How are you doing my friend?','morsecode.wav', 10, 440,Fs,20),
    #   (to play the file, and make the length 2^20)
    #   x = morse('How are you doing my friend?','morsecode.wav', 3, 440,Fs, 20, 2^20,True),
    #
    #   Copyright 2018 Mauri Niininen, AG1LE
    """

    # t = 0:1/Fs:1.2/code_speed,  #One dit of time at w wpm is 1.2/w.

    t = np.linspace(
        0.0,
        1.2 / code_speed,
        num=int(Fs * 1.2 / code_speed),
        endpoint=True,
        retstep=False,
    )

    Dit = np.sin(2 * np.pi * f_code * t)
    ssp = np.zeros(len(Dit))
    # one Dah of time is 3 times  dit time
    t2 = np.linspace(
        0.0,
        3 * 1.2 / code_speed,
        num=3 * int(Fs * 1.2 / code_speed),
        endpoint=True,
        retstep=False,
    )
    # Dah = np.concatenate((Dit,Dit,Dit))
    Dah = np.sin(2 * np.pi * f_code * t2)

    # lsp = (np.zeros(len(Dah)),)  # changed size argument to function of Dah

    # Defining Characters & Numbers
    Codebook = {
        "A": np.concatenate((Dit, ssp, Dah)),
        "B": np.concatenate((Dah, ssp, Dit, ssp, Dit, ssp, Dit)),
        "C": np.concatenate((Dah, ssp, Dit, ssp, Dah, ssp, Dit)),
        "D": np.concatenate((Dah, ssp, Dit, ssp, Dit)),
        "E": Dit,
        "F": np.concatenate((Dit, ssp, Dit, ssp, Dah, ssp, Dit)),
        "G": np.concatenate((Dah, ssp, Dah, ssp, Dit)),
        "H": np.concatenate((Dit, ssp, Dit, ssp, Dit, ssp, Dit)),
        "I": np.concatenate((Dit, ssp, Dit)),
        "J": np.concatenate((Dit, ssp, Dah, ssp, Dah, ssp, Dah)),
        "K": np.concatenate((Dah, ssp, Dit, ssp, Dah)),
        "L": np.concatenate((Dit, ssp, Dah, ssp, Dit, ssp, Dit)),
        "M": np.concatenate((Dah, ssp, Dah)),
        "N": np.concatenate((Dah, ssp, Dit)),
        "O": np.concatenate((Dah, ssp, Dah, ssp, Dah)),
        "P": np.concatenate((Dit, ssp, Dah, ssp, Dah, ssp, Dit)),
        "Q": np.concatenate((Dah, ssp, Dah, ssp, Dit, ssp, Dah)),
        "R": np.concatenate((Dit, ssp, Dah, ssp, Dit)),
        "S": np.concatenate((Dit, ssp, Dit, ssp, Dit)),
        "T": Dah,
        "U": np.concatenate((Dit, ssp, Dit, ssp, Dah)),
        "V": np.concatenate((Dit, ssp, Dit, ssp, Dit, ssp, Dah)),
        "W": np.concatenate((Dit, ssp, Dah, ssp, Dah)),
        "X": np.concatenate((Dah, ssp, Dit, ssp, Dit, ssp, Dah)),
        "Y": np.concatenate((Dah, ssp, Dit, ssp, Dah, ssp, Dah)),
        "Z": np.concatenate((Dah, ssp, Dah, ssp, Dit, ssp, Dit)),
        ".": np.concatenate((Dit, ssp, Dah, ssp, Dit, ssp, Dah, ssp, Dit, ssp, Dah)),
        ",": np.concatenate((Dah, ssp, Dah, ssp, Dit, ssp, Dit, ssp, Dah, ssp, Dah)),
        "?": np.concatenate((Dit, ssp, Dit, ssp, Dah, ssp, Dah, ssp, Dit, ssp, Dit)),
        "/": np.concatenate((Dah, ssp, Dit, ssp, Dit, ssp, Dah, ssp, Dit)),
        "=": np.concatenate((Dah, ssp, Dit, ssp, Dit, ssp, Dit, ssp, Dah)),
        "1": np.concatenate((Dit, ssp, Dah, ssp, Dah, ssp, Dah, ssp, Dah)),
        "2": np.concatenate((Dit, ssp, Dit, ssp, Dah, ssp, Dah, ssp, Dah)),
        "3": np.concatenate((Dit, ssp, Dit, ssp, Dit, ssp, Dah, ssp, Dah)),
        "4": np.concatenate((Dit, ssp, Dit, ssp, Dit, ssp, Dit, ssp, Dah)),
        "5": np.concatenate((Dit, ssp, Dit, ssp, Dit, ssp, Dit, ssp, Dit)),
        "6": np.concatenate((Dah, ssp, Dit, ssp, Dit, ssp, Dit, ssp, Dit)),
        "7": np.concatenate((Dah, ssp, Dah, ssp, Dit, ssp, Dit, ssp, Dit)),
        "8": np.concatenate((Dah, ssp, Dah, ssp, Dah, ssp, Dit, ssp, Dit)),
        "9": np.concatenate((Dah, ssp, Dah, ssp, Dah, ssp, Dah, ssp, Dit)),
        "0": np.concatenate((Dah, ssp, Dah, ssp, Dah, ssp, Dah, ssp, Dah)),
        " ": np.concatenate((ssp, ssp, ssp, ssp, ssp, ssp, ssp)),  # 7 dits
    }
    text = text.upper()

    # dit duration in seconds
    dit = 1.2 / code_speed
    # calculate the length of text in dit units
    txt_dits = MorseCode(text).len
    # calculate total text length in seconds
    tot_len = txt_dits * dit

    # assert length_seconds - tot_len > 0
    if length_seconds - tot_len < 0:
        print(text + str(" too long. Will not be generated."))
        return np.zeros([1, 1])

    # calculate how many dits will fit in the
    pad_dits = int((length_seconds - tot_len) / dit)

    # pad with random space to fit proper length
    morsecode = []
    pad = random.randint(0, pad_dits)
    # print("pad_dits:{} pad:{}".format(pad_dits,pad))
    for i in range(pad):
        morsecode = np.concatenate((morsecode, ssp))

    # start with pause (7 dit lengths)
    # morsecode= np.concatenate((ssp,ssp,ssp,ssp,ssp,ssp,ssp))

    # concatenate all characters
    for ch in text:
        if ch == " ":
            morsecode = np.concatenate((morsecode, ssp, ssp, ssp, ssp))
        elif ch == "\n":
            pass
        else:
            val = Codebook[ch]
            morsecode = np.concatenate((morsecode, val, ssp, ssp, ssp))

    # morsecode = np.concatenate((morsecode, lsp))

    if total_seconds:
        append_length = Fs * total_seconds - len(morsecode)
        if append_length < 0:
            print(
                "Length {} isn't large enough for your message, it must be > {}.\n".format(
                    append_length, len(morsecode)
                )
            )
            return morsecode
        else:
            morsecode = np.concatenate((morsecode, np.zeros(append_length)))

    # end with pause (14 dit lengths)
    morsecode = np.concatenate(
        (
            morsecode,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
            ssp,
        )
    )

    # noise = randn(size(morsecode)),
    # [noisy,noise] = addnoise(morsecode,noise,snr),

    if SNR_dB:
        # https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal
        # Desired SNR in dB

        # Desired linear SNR
        SNR_linear = 10.0 ** (SNR_dB / 10.0)
        # print( "Linear snr = ", SNR_linear)

        # Measure power of signal - assume zero mean
        power = morsecode.var()
        # print ("Power of signal = ", power)

        # Calculate required noise power for desired SNR
        noise_power = power / SNR_linear
        # print ("Noise power = ", noise_power )
        # print ("Calculated SNR = {:4.2f} dB".format(10*np.log10(power/noise_power )))

        # Generate noise with calculated power (mu=0, sigma=1)
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(morsecode))

        # Add noise to signal
        morsecode = noise + morsecode

    # Normalize before saving
    max_n = (max(morsecode),)
    morsecode = morsecode * max_n * 10000

    if file_name:
        # TODO - Cannot write "?","/" to file
        # write(file_name, Fs, morsecode)
        write(file_name, Fs, morsecode.astype(np.int16))
    if play_sound:
        # deprecated
        # sd.play(morsecode, Fs)
        pass
    return morsecode
