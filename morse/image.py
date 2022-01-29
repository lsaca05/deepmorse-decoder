import cv2
import pyaudio
from scipy.io import wavfile
from find_peak import find_peak
from process_audio_file import process_audio_file

def create_image2(filename, imgSize, dataAugmentation=False):

    RATE = 8000
    FORMAT = pyaudio.paInt16  # conversion format for PyAudio stream
    CHANNELS = 1  # microphone audio channels
    CHUNK_SIZE = 8192  # number of samples to take per read
    SAMPLE_LENGTH = int(CHUNK_SIZE * 1000 / RATE)  # length of each sample in ms
    SAMPLES_PER_FRAME = 4 

    rate, data = wavfile.read(filename)
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
