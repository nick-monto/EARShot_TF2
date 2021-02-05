import tensorflow as tf
import pandas as pd

from earshot.audio import *
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

path_to_plk = '../training_1k.plk'
path_to_wav = '../EARSHOT_NEW_WAVS' # needed to generate category dict

try:
    training_1k_df = pd.read_pickle(path_to_plk)
except:
    print('Pickle file possibly saved on python version >= 3.8')
    print('Attempting to load file with pickle5.')
    # need to load on python <= 3.7 in order to load pickles saved in 3.8
    import pickle5 as pickle
    with open(path_to_plk, 'rb') as fh:
        training_1k_df = pickle.load(fh)

p = ModelParameters()

model = EARSHOT(len(training_1k_df['Target'].iloc()[0]), p)
model.compile(loss=model.loss, optimizer="adam", metrics=[tf.keras.metrics.CategoricalAccuracy()], run_eagerly=False)

# Example with the words RIB and RICH with no ISI
pair_predictions = PairPredictions('RIB','RICH', None, training_1k_df,
                                   model,'./earshot/checkpoints/cp-06000.ckpt')

df_list = []
for key in list(pair_predictions.cosine_dict.keys()):
    df_list.append(pair_predictions.cosine_dict[key][['RIB','RICH']])

# average length of RIB
pair_predictions.word1_df['Input'].iloc()[2].shape
mean_length_w1 = int(sum(len(a) for a in pair_predictions.word1_df['Input'].tolist()) / len(pair_predictions.word1_df['Input']))

pd.concat(df_list).groupby(level=0).mean().plot.line(xlabel='Time Steps',ylabel='Cosine Simularity', ylim=(0,1), title='RIB -> RICH')
plt.axvline(mean_length_w1, color='black', linestyle='--') # line for average word 1 length
plt.show()

# Example with the words RIB and RICH with a 25 step ISI
pair_predictions = PairPredictions('RIB','RICH', 25, training_1k_df,
                                   model,'./earshot/checkpoints/cp-06000.ckpt')

df_list = []
for key in list(pair_predictions.cosine_dict.keys()):
    df_list.append(pair_predictions.cosine_dict[key][['RIB','RICH']])

pair_predictions.word1_df['Input'].iloc()[2].shape
mean_length_w1 = int(sum(len(a) for a in pair_predictions.word1_df['Input'].tolist()) / len(pair_predictions.word1_df['Input']))

pd.concat(df_list).groupby(level=0).mean().plot.line(xlabel='Time Steps',ylabel='Cosine Simularity', ylim=(0,1), title='RIB -> RICH')
plt.axvline(mean_length_w1, color='black', linestyle='--') # line for average word 1 length
plt.axvline(mean_length_w1+25, color='black', linestyle='--') # line for average word 1 length

plt.show()
