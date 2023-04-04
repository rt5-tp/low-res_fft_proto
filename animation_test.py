


# Import relevant libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# In[8]:


frame = np.zeros((16,32,3),dtype=np.uint8) # initialise a frame

srate,data = wavfile.read('racingrats.wav') # Load audio



class SignalPlotter:
    def __init__(self,srate,data):
        self.srate, self.data = srate,data
        self.duration = len(self.data)/self.srate
        self.t = np.arange(0, self.duration,1/self.srate)
        self.data_norm = self.data/(max(np.amax(self.data), -np.amin(self.data)))

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

plotter = SignalPlotter(srate,data)

x_plotter = np.linspace(20, plotter.srate/2, len(plotter.freqSignal()))



fig3, (ax1, ax2) = plt.subplots(1,2)
fig3.suptitle("Test Audio Plot")
freq_plot = 10*np.log10(np.abs(plotter.freqSignal()))
ax1.plot(x_plotter, freq_plot)
ax2.plot(plotter.t, plotter.normSignal())

ax1.set_xscale('log')
# ax2.set_xscale('log')
ax1.set_xlabel("Frequency, Hz")
ax1.set_ylabel("Magnitude, dB")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Normalised Amplitude")
ax1.set_title("Frequency")
ax2.set_title("Normalised Amplitude")
plt.show()


print(len(data[:,0]))



freq_spectrum = np.abs(plotter.freqSignal())[:,0]
length = len(freq_spectrum)
bins = np.linspace(0,length,33)

colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))
colour_bins = np.linspace(0,length,4)

print('freq_spectrum:',freq_spectrum)
print('length:',length)
print('bins:',bins)

print('colour_spectrum:', colour_spectrum)
print('colour_bins:', colour_bins)

mask = np.zeros((16,32,3),dtype=np.uint8)

mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum)) # scale the y-axis to range between 0 and 16

for i in range(len(bins)-1):
    bin = np.amax(mask_spectrum[int(bins[i]):int(bins[i+1])])
    mask[16-int(bin):,i,:] = 255
    
frame = np.zeros((16,32,3),dtype=np.uint8)

colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))

colour_bins = np.linspace(0,length,3)

for i in range(len(colour_bins)-1):
    bin = np.amax(colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
    frame[:,:,i] = int(bin)

frame[mask==0] = 0

plt.imshow(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
plt.show



Fs_audio = srate
Fs_video  = 30

out_path = "Test_out.mp4"

n_frame = Fs_audio//Fs_video # Number of audio samples per frame

n_frames = len(data)//n_frame

print(n_frame)


for i in range(2):
    ptr = i*(n_frame)
    buffer = data[ptr:ptr+n_frame,0]
    print(len(buffer))


for i in range(2):
    ptr = i*(n_frame)
    buffer = data[ptr:ptr+n_frame,:]

    plotter = SignalPlotter(Fs_audio,buffer)

    x_plotter = np.linspace(20, plotter.srate/2, len(plotter.freqSignal()))

    fig3, (ax1, ax2) = plt.subplots(1,2)
    fig3.suptitle("Test Buffer "+str(i)+" Plot")
    freq_plot = 10*np.log10(np.abs(plotter.freqSignal()))
    ax1.plot(x_plotter, freq_plot)
    ax2.plot(plotter.t, plotter.normSignal())

    ax1.set_xscale('log')
    # ax2.set_xscale('log')
    ax1.set_xlabel("Frequency, Hz")
    ax1.set_ylabel("Magnitude, dB")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalised Amplitude")
    ax1.set_title("Frequency")
    ax2.set_title("Normalised Amplitude")
    plt.show()
    
 

for i in range(2):
    ptr = i*(n_frame)
    buffer = data[ptr:ptr+n_frame,:]

    plotter = SignalPlotter(Fs_audio,buffer)

    freq_spectrum = np.abs(plotter.freqSignal())[:,0]
    length = len(freq_spectrum)

    bins = np.linspace(0,length,33)
    colour_bins = np.linspace(0,length,4)

    colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))
    mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum)) # scale the y-axis to range between 0 and 16

    mask = np.zeros((16,32,3),dtype=np.uint8)
    frame = np.zeros((16,32,3),dtype=np.uint8)

    for j in range(len(bins)-1):
        bin = np.amax(mask_spectrum[int(bins[j]):int(bins[j+1])])
        mask[16-int(bin):,j,:] = 255

    

    for i in range(len(colour_bins)-1):
        bin = np.amax(colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
        frame[:,:,i] = int(bin)

    frame[mask==0] = 0

    plt.figure()
    plt.imshow(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    plt.show

    

out = cv.VideoWriter(out_path,cv.VideoWriter.fourcc(*'mp4v'),Fs_video,(32,16))

for i in range(n_frames):
    ptr = i*(n_frame)
    buffer = data[ptr:ptr+n_frame,:]

    plotter = SignalPlotter(Fs_audio,buffer)

    freq_spectrum = np.abs(plotter.freqSignal())[:,0]
    length = len(freq_spectrum)

    bins = np.linspace(0,length,33)
    colour_bins = np.linspace(0,length,4)

    colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))
    mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum)) # scale the y-axis to range between 0 and 16

    mask = np.zeros((16,32,3),dtype=np.uint8)
    frame = np.zeros((16,32,3),dtype=np.uint8)

    for j in range(len(bins)-1):
        bin = np.amax(mask_spectrum[int(bins[j]):int(bins[j+1])])
        mask[16-int(bin):,j,:] = 255

    

    for i in range(len(colour_bins)-1):
        bin = np.amax(colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
        frame[:,:,i] = int(bin)

    frame[mask==0] = 0

    out.write(frame)

out.release()
cv.destroyAllWindows()

    


Fs_video = 12
n_frame = Fs_audio//Fs_video # Number of audio samples per frame
n_frames = len(data)//n_frame

out = cv.VideoWriter("12fps.mp4",cv.VideoWriter.fourcc(*'mp4v'),Fs_video,(32,16))

for i in range(n_frames):
    ptr = i*(n_frame)
    buffer = data[ptr:ptr+n_frame,:]

    plotter = SignalPlotter(Fs_audio,buffer)

    freq_spectrum = np.abs(plotter.freqSignal())[:,0]
    length = len(freq_spectrum)

    bins = np.linspace(0,length,33)
    colour_bins = np.linspace(0,length,4)

    colour_spectrum = freq_spectrum*(255/np.amax(freq_spectrum))
    mask_spectrum = freq_spectrum*(16/np.amax(freq_spectrum)) # scale the y-axis to range between 0 and 16

    mask = np.zeros((16,32,3),dtype=np.uint8)
    frame = np.zeros((16,32,3),dtype=np.uint8)

    for j in range(len(bins)-1):
        bin = np.amax(mask_spectrum[int(bins[j]):int(bins[j+1])])
        mask[16-int(bin):,j,:] = 255

    

    for i in range(len(colour_bins)-1):
        bin = np.amax(colour_spectrum[int(colour_bins[i]):int(colour_bins[i+1])])
        frame[:,:,i] = int(bin)

    frame[mask==0] = 0

    out.write(frame)

out.release()
cv.destroyAllWindows()

    
