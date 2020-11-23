import librosa
import math
import os
import pandas as pd
import numpy.matlib as npm
import matplotlib.pyplot as plt

from earshot.audio import *
from numpy import ones
from scipy import sparse, fftpack, spatial
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

# TODO include a pass for precomputed input to the Batch Generator / Manifest

def pad(data_2d, ref_shape, pad_val=-9999):
    '''
    Pads a 2D matrix "data" with the pad_value to make it have shape ref_shape.
    All the padded values occur "off the end" of the data array, so that the
    original array occupies the upper left of the resulting matrix.

    Refer to pad_nans method to pad with nan values
    '''
    padded_data = pad_val*ones(ref_shape)
    padded_data[:data_2d.shape[0], :data_2d.shape[1]] = data_2d
    return padded_data


def spectro_calc(audio_path):
    # TODO add pass for parameters
    sig = librosa.core.load(audio_path, sr=22050)[0]
    sig = librosa.effects.trim(sig, frame_length=32, hop_length=16)[0]
    spec = spectrogram(sig, samp_rate=22050, dimension=256,
                       frame_len=10, frame_shift=10)
    return np.transpose(spec).astype(np.float32)


class Manifest(object):
    '''
    Generates data manifest when passed a folder containing audio files with
    naming structure of word_talker.wav
    '''
    def __init__(self, audio_dir):
        '''
        audio_dir = path to directory containing desired audio files
        '''
        path_list = list()
        file_list = list()
        for (dirpath, dirnames, filenames) in os.walk(audio_dir):
            path_list += [os.path.join(dirpath, file) for file in filenames if (
                file.endswith('.wav') or file.endswith('.WAV'))]
            file_list += [file for file in filenames if (
                file.endswith('.wav') or file.endswith('.WAV'))]
        manifest = []
        for i in range(len(file_list)):
            word, talker = file_list[i][:-4].split('_')
            manifest.append([path_list[i], word, talker])
        self.manifest = pd.DataFrame(
            manifest, columns=['Path', 'Word', 'Talker'])
        uniques = self.manifest.nunique()
        print("Your dataset contains {} unique word(s) and {} unique talker(s).".format(
              uniques[1], uniques[2]))
        print("\n")
        print("Number of utterances by a unique talker.")
        print(self.manifest.groupby('Word').count()['Talker'])
        print("\n")
        print("Number of unique word utterances by each talker.")
        print(self.manifest.groupby('Talker').count()['Word'])

    def generate_srvs(self, target='Word', target_len=300, target_on=10):
        '''
        target = manifest column containing desired targets
        target_len = desired length of target vector
        target_on = desired number of ones in target vector
        '''
        assert target in [
            'Talker', 'Word'], "Please use 'Word' or 'Talker' as your target."
        target_items = self.manifest[target].unique()
        target_pairs = []
        for i in range(len(target_items)):
            target_pairs.append([target_items[i],
                                 sparse.random(1, target_len,
                                               density=target_on/target_len, data_rvs=ones).toarray()[0]])
        target_pairs = pd.DataFrame(target_pairs, columns=[target, 'Target'])
        print("There are {} unique target patterns.".format(
            len(target_pairs['Target'])))
        self.manifest = self.manifest.merge(target_pairs)

    def calc_spec(self):
        # place holder for input
        self.manifest['Input'] = None
        for index,row in tqdm(self.manifest.iterrows()):
            self.manifest.loc[index, 'Input'] = spectro_calc(
                self.manifest['Path'][index])

    def gen_predict(self, Talker, num_samp=100, pad_value=-9999, window_len=0.035, skip_len=0.005, max_freq=8000):
        # pull a number of random items from a specific talker for model prediction
        predict_df = self.manifest[self.manifest.Talker == Talker].sample(num_samp)
        
        predict_df['Input'] = None
        for index, row in tqdm(predict_df.iterrows()):
            predict_df.loc[index, 'Input'] = AudioTools(predict_df['Path'][index]).sgram(window_len,skip_len,max_freq)
        M = max(len(a) for a in predict_df['Input'].tolist())
        
        predict_df['Padded Input'] = None
        for index, row in predict_df.iterrows():
            s = predict_df.loc[index, 'Input']
            predict_df.loc[index, 'Padded Input'] = pad(s, (M, s.shape[1]), pad_val=pad_value)
        return predict_df



class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, df, batch_size=32, pad_value=-9999, return_seq=True, spec_calc='scipy'):
        '''
        df = manifest dataframe
        batch_size = desired batching size
        pad_value = desired value to pad data with, pads to max of each batch
        return_seq = determines generated label structure, related to model architecture
        '''
        assert spec_calc in [
            'scipy', 'librosa'], "Please use 'scipy' or 'librosa' to calculate spectrograms."
        self._spec_calc = spec_calc
        self.df = df
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.targets = np.array(self.df['Target'].tolist()).astype(np.float32)
        self.path_list = self.df['Path'].tolist()
        self.indexes = np.arange(len(self.path_list))
        self.return_seq = return_seq

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_pairs_temp = []
        for k in indexes:
            list_pairs_temp.append((self.path_list[k], self.targets[k]))

        # Generate data
        X, y = self.__data_generation(list_pairs_temp)

        return X, y

    def __data_generation(self, list_pairs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        # calculate spectrograms for each item in batch
        if self._spec_calc == 'librosa':
            spec = [ spectro_calc(path[0]) for path in list_pairs_temp ]
        else:
            spec = [ AudioTools(path[0]).sgram(0.035,0.005,8000) for path in list_pairs_temp ]
        # get max len of batch
        M = max(len(a) for a in spec)
        # pad all specs in batch to max length
        padded_spec = [ pad(s, (M, s.shape[1]), pad_val=self.pad_value)
                       for s in spec ]
        X = np.empty((self.batch_size, M, spec[0].shape[1]), dtype=np.float32)
        # pad targets if LSTM layer has return_sequence=True
        if self.return_seq:
            y = np.empty((self.batch_size, M, len(list_pairs_temp[0][1])), dtype=np.float32)
            targets = [ npm.repmat(pair[1],spec[i].shape[0],1) for i,pair in enumerate(list_pairs_temp) ]
            padded_targets = [ pad(i,(M,i.shape[1]),pad_val=self.pad_value) for i in targets ]
            for i, pair in enumerate(list_pairs_temp):
                X[i, ] = padded_spec[i]
                y[i, ] = padded_targets[i]
        else:
            y = np.empty((self.batch_size, len(list_pairs_temp[0][1])), dtype=np.float32)

            for i, pair in enumerate(list_pairs_temp):
                X[i, ] = padded_spec[i]
                y[i] = pair[1]

        return X, y

class AudioTools(object):
    '''
    A suite of tools to generate SPL spectrograms for a specified audio file.

    The reference for the majority of these functions can be found here:
    https://courses.engr.illinois.edu/ece590sip/sp2018/spectrograms1_wideband_narrowband.html

    '''
    def __init__(self, audiopath):
        '''
        audiopath: path to desired audio
        '''
        sig, self.fs = librosa.core.load(audiopath)
        self.signal = librosa.effects.trim(
            sig, frame_length=32, hop_length=16)[0]

    def _enframe(self, signal, skip_len, window_len):
        # generate time windows for subsequent stft calculations
        # w = 0.54*np.ones(L)
        # for n in range(0,L):
        #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
        w = np.hamming(window_len)
        frames = []
        nframes = 1+int((len(signal)-window_len)/skip_len)
        for t in range(0, nframes):
            frames.append(
                np.copy(signal[(t*skip_len):(t*skip_len+window_len)])*w)
        return frames

    def _stft(self, frames, n_fft, Fs):
        stft_frames = [fftpack.fft(x, n_fft) for x in frames]
        freq_axis = np.linspace(0, Fs, n_fft)
        return(stft_frames, freq_axis)

    def _stft2level(self, stft_spectra, max_freq_bin):
        magnitude_spectra = [np.abs(x) for x in stft_spectra]
        max_magnitude = max([max(x) for x in magnitude_spectra])
        min_magnitude = max_magnitude / 1000.0
        for t in range(0, len(magnitude_spectra)):
            for k in range(0, len(magnitude_spectra[t])):
                magnitude_spectra[t][k] /= min_magnitude
                if magnitude_spectra[t][k] < 1:
                    magnitude_spectra[t][k] = 1
        # convert to SPL spectra
        level_spectra = [20*np.log10(x[0:max_freq_bin])
                         for x in magnitude_spectra]
        return level_spectra

    def sgram(self, window_len, skip_len, max_freq):
        '''
        window_len: length of window in seconds
        skip_len: length to skip in seconds
        max_freq: maximum desired frequency
        '''
        self.n_fft = pow(2, int(math.log(int(self.fs*window_len),
                                         2) + 0.5))  # calc NFFT suitable for window_len
        window_len = int(window_len*self.fs)  # convert to length in samples
        skip_len = int(skip_len*self.fs)  # convert to length in samples
        self.frames = self._enframe(self.signal, skip_len, window_len)
        spectra, self.freq_axis = self._stft(self.frames, self.n_fft, self.fs)
        self.spl_spec = np.array(self._stft2level(
            spectra, int(max_freq*self.n_fft/self.fs)))
        self.max_time = len(self.frames)*skip_len/self.fs
        self.max_freq = max_freq
        # returns max scaled spl spectrogram
        return self.spl_spec / self.spl_spec.max()

    def plot_spec(self):
        try:
            plt.imshow(self.spl_spec.transpose(),
                       origin='lower',
                       extent=(0, self.max_time, 0, self.max_freq),
                       aspect='auto')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()
        except:
            print("Please run the sgram method before plotting.")

class Prediction(object):
    '''
    Class for generating cosine simularity dfs.
    '''
    def __init__(self, trained_model, prediction_df, full_manifest):
        '''
        trained_model: trained earshot model
        prediction_set: set of data for prediction, easily generated using gen_predict method of the Manifest class
        full_manifest: full manifest df from Manifest class (.manifest method) that contains full set of words and targets
        '''
        self.prediction_df = prediction_df


        self.predictions = trained_model.predict(np.array(self.prediction_df['Padded Input'].tolist()),
                                                 batch_size = 32,
                                                 verbose=1)

        # creat df of unique words with associated labels
        self.unique_label_df = full_manifest[~full_manifest['Word'].duplicated()][['Word','Target']].reset_index()

        # get talker, word, and input of each input
        self.true_items = [ (row['Talker'], row['Word'], row['Input']) for index, row in self.prediction_df.iterrows() ]
        
        self.cosine_sim_dict = self._calc_cosine()

    def _calc_cosine(self):
            
        # calculate cosine similarity of each prediction
        # at each time step
        # to every word in the training vocab
        # constrained to the true length of the prediction input

        # compute consine simularity at each timestep against each word in vocab
        # output rows = timestep, columns = word, cell value = consine simularity
        # this method currently overrides duplicate targets in dictionary with most recent in set
        simularity_dict = {}
        for i in tqdm(range(len(self.true_items))):
            simularity_df = pd.DataFrame()
            for index, row in self.unique_label_df.iterrows():
                cosine = []
                for t in range(len(self.true_items[i][2])):
                    # compute cosine simularity
                    cosine.append(1 - spatial.distance.cosine(row['Target'], self.predictions[i][t]))
                # make dataframe containing simularity computed at each time step to every word
                simularity_df[row['Word']] = cosine 
            # add df to dict with key of the true word
            simularity_dict["{0}".format(self.true_items[i][1])] = simularity_df
        return simularity_dict