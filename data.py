import os
import time
from os.path import isdir, join
from tqdm import tqdm
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sounddevice as sd


# Plot log-specgram
def plot_log_spectrogram(freqs, times, spectrogram, sample_rate=16000, title='Log Spectrogram'):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Wav')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title(title)
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')
    plt.show()


def play_sound(data_class='bird', idx=None, wav_path="data/train/audio"):
    class_path = join(wav_path, data_class)
    class_wavs = sorted([wav for wav in os.listdir(class_path)])
    if idx:
        sd.play(class_dir[idx])
    else:
        print('play all ', data_class)
        for wavpath in class_wavs:
            sr, samples = wavfile.read(join(class_path,wavpath))
            sd.play(samples)
            time.sleep(1)


def plot_spec3d(spectrogram):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(spectrogram.shape[1]),
                       np.arange(spectrogram.shape[0]))  # freq in rows, time in columns
    surf = ax.plot_surface(X, Y, spectrogram,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(spectrogram.min(), spectrogram.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    ax.set_zlabel('Z (values)')
    plt.show()


# Reconstruction
def griffin_lim(spec):
    # ability to listen to average spectrograms
    pass


def plot_means(means, rows=12):
    rows = len(means.keys())
    # plt.figure(figsize=(14, 4))  # size in inches
    i = 1
    for (k, v) in means.items():
        print(k)
        plt.subplot(rows, 2, i)
        i +=1
        plt.title('Mean fft of ' + k)
        plt.plot(v['fft_mean'])
        plt.grid()
        plt.subplot(rows, 2, i)
        i +=1
        plt.title('Mean specgram of ' + k)
        plt.imshow(v['spec_mean'].T, aspect='auto', origin='lower')
        # extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        # plt.yticks(freqs[::16])
        # plt.xticks(times[::16])
    plt.tight_layout()
    plt.show()


def demo():
    train_audio_path = "/home/erik/Audio/tf-speech/data/train/audio/"
    audio_file = join(train_audio_path, "yes/0a7c2a8d_nohash_0.wav")
    sr, samples = wavfile.read(audio_file)  # 16kHz
    sd.default.samplerate = sr

    # Extract spectrogram
    freqs, times, spectrogram = log_specgram(samples, sr)

    # Plot
    plot_log_spectrogram(freqs, times, spectrogram)
    plot_spec3d(spectrogram)

    # Normalise 
    spec_max = np.max(spectrogram, axis=0)
    spec_min = np.min(spectrogram, axis=0)
    spec_mean = np.mean(spectrogram, axis=0)
    spec_std = np.std(spectrogram, axis=0)
    spectrogram = (spectrogram - spec_mean) / spec_std

    plot_spec3d(spectrogram)

    number_of_recordings = []
    for d in dirs:
        waves = [f for f in os.listdir(join(train_audio_path, d)) if f.endswith('.wav')]
        number_of_recordings.append(len(waves))

    plt.bar(x=np.arange(len(dirs)), height=number_of_recordings)
    plt.show()


class DatasetVisualizer(object):
    def __init__(self, train_audio_path="data/train/audio/",
                 window_size=20, step_size=10):
        self.train_audio_path = train_audio_path
        self.classes = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
        self.classes.sort()
        self.classes = self.classes[1:]

        # creates self.means: dict
        # self.means['class']['spec_mean'], self.means['class']['fft_mean']
        # self._get_all_spec_fft_mean()  

        self.window_size = window_size
        self.step_size = step_size

    def log_spectrogram(self, audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    def custom_fft(self, y, fs):
        T = 1.0 / fs
        fft_len = int(50 * fs / 1e3)
        N = y.shape[0]
        yf = fft(y, n=fft_len )
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
        # FFT is also complex, to we take just the real part (abs)
        return xf, vals

    def extract_mean_fft_spectrogram(self, _class, verbose=False):
        fft_values = []
        spectrograms = []
        waves = [f for f in os.listdir(join(self.train_audio_path, _class)) if f.endswith('.wav')]
        if verbose:
            print('{}', _class)
            print('samples: ', len(waves))
        for wav in waves:
            sample_rate, samples = wavfile.read(self.train_audio_path + _class + '/' + wav)
            if samples.shape[0] != 16000:
                continue
            xf, vals = self.custom_fft(samples, 16000)
            fft_values.append(vals)
            freqs, times, spec = self.log_spectrogram(samples, 16000)
            spectrograms.append(spec)
        fft_values_mean = np.mean(np.array(fft_values), axis=0)
        spectrograms_mean = np.mean(np.array(spectrograms), axis=0)
        return spectrograms_mean, fft_values_mean

    def calc_means(self):
        means = {}
        for c in tqdm(self.classes, desc='Extracting means'):
            spec_mean, fft_mean = self.extract_mean_fft_spectrogram(c)
            means[c] = {'spec_mean': spec_mean, 'fft_mean': fft_mean}
        self.means = means
        return means

    def load_means(self):
        self.means = np.load(join(self.train_audio_path, 'means.npy')).item()

    def plot_mean(self, category='yes'):
        mean = self.means[category]
        plt.figure(figsize=(14,4))
        plt.subplot(1, 2, 1)
        plt.title('Mean fft of ' + category)
        plt.plot(mean['fft_mean'])
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.title('Mean spectrogram of ' + category)
        plt.imshow(mean['spec_mean'].T, aspect='auto', origin='lower',
                   extent=[0, 1, 20, 8000])
        plt.show()

    def plot_mean3D(self, category='yes'):
        spectrogram = self.means[category]['spec_mean']

        fig = plt.figure(figsize=(10,10))
        plt.suptitle('Average spectrogram: '+category)
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(spectrogram.shape[0]),
                           np.arange(spectrogram.shape[1]))  # freq in rows, time in columns
        surf = ax.plot_surface(X, Y, spectrogram.T,
                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(spectrogram.min(), spectrogram.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('frequencies')
        ax.set_zlabel('magnitude')
        plt.show()



if __name__ == "__main__":

    data = DatasetVisualizer()
    data.calc_means()
    data.load_means()
    print(data.means.keys())
    data.plot_mean('yes')
    data.plot_mean3D('yes')



