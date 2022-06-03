# deepmorse-decoder
Deep Learning Morse Decoder created by Mauri Niininen (AG1LE).
Further improved by Lucas Saca.


## Getting Started

Make sure you have Python 3.6.5 or later available in your system.

Downloading and utilizing cuDNN will make model training significantly faster --  took me about 1.25-1.5 hours for the model to converge. TF2 requires CUDA v11.x.

To use the spectrogram in Python 3.7+, download the appropriate unofficial PyAudio .whl file from [UCI's Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install PyAudio from it.

Clone this GitHub repository, and create a virtual Python environment.
Install Python libraries using requirements.txt file

```bash
git clone https://github.com/ag1le/deepmorse-decoder.git
cd deepmorse-decoder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start the program

```bash
python specgram.py
````

You should see the program starting and a spectrogram display should pop up.
The program is listening your microphone using pyaudio library.
You can play an audio source with Morse code and you should now see the 4 second buffer in the spectrogram display.

![GitHub Logo](/images/screen.png)
Format: ![Alt Text](url)

## The CNN-LSTM-CTC model
The model files are stored in the model-arrl#/ directory.
You can create or retrain the model using the MorseDecoder.py in the morse/ directory.
For instructions you can use the --help option.

```bash
python morse/MorseDecoder.py -h 
```
