import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


class video():
    def __init__(self, shape, frate):
        frame = np.zeros((16, 32, 3), dtype=np.uint8)


class SignalPlotter:
    def __init__(self, srate, data):
        self.srate, self.data = srate, data
        self.duration = len(self.data)/self.srate
        self.t = np.arange(0, self.duration, 1/self.srate)
        self.data_norm = self.data / \
            (max(np.amax(self.data), -np.amin(self.data)))

    def rawSignal(self):
        '''Returns an array for the raw signal'''
        return self.data

    def normSignal(self):
        '''Returns an array for the normalised signal'''
        return self.data_norm

    def freqSignal(self):
        '''Performs fft on normalised signal and returns array'''
        data_norm_f = np.fft.fft(self.data_norm)
        data_norm_f = data_norm_f[:len(data_norm_f)//2]
        return data_norm_f


srate, data = wavfile.read('racingrats.wav')

plotter = SignalPlotter(srate, data)

x_plotter = np.linspace(20, plotter.srate/2, len(plotter.freqSignal()))

freq_spectrum = np.abs(plotter.freqSignal())
length = len(freq_spectrum)

bins = np.linspace(0, length, 33)

mask = np.zeros((16, 32, 3), dtype=np.uint8)

# scale the y-axis to range between 0 and 16
mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum))

for i in range(len(bins)-1):
    bin = np.amax(mask_spectrum[int(bins[i]):int(bins[i+1])])
    mask[16-int(bin):, i, :] = 255

frame = np.zeros((16, 32, 3), dtype=np.uint8)

colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))

colour_bins = np.linspace(0, 716981, 4)

for i in range(len(bins)-1):
    bin = np.amax(colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
    frame[:, :, i] = int(bin)

frame[mask == 0] = 0

out = cv.VideoWriter('video.mp4', cv.VideoWriter.fourcc(*'mp4v'), 20, (16, 32))

n_frame = srate//20

number_of_frames = len(data)//n_frame

for i in range(number_of_frames):
    pntr = i*number_of_frames
    buffer = data[pntr:pntr+number_of_frames]

    if len(buffer) < n_frame:
        break

    buffer_plot = SignalPlotter(buffer)

    bins = np.linspace(0, length, 33)

    mask = np.zeros((16, 32, 3), dtype=np.uint8)

    # scale the y-axis to range between 0 and 16
    mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum))

    for i in range(len(bins)-1):
        bin = np.amax(mask_spectrum[int(bins[i]):int(bins[i+1])])
        mask[16-int(bin):, i, :] = 255

    colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))

    colour_bins = np.linspace(0, 716981, 4)

    for i in range(len(bins)-1):
        bin = np.amax(
            colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
        frame[:, :, i] = int(bin)
        frame[mask==0] = 0

    out.write(frame)

out.release()
cv.destroyAllWindows()
