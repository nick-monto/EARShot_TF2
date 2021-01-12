import tensorflow as tf
import pandas as pd

from earshot.audio import *
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

path_to_plk = 'path to pickled training data'
path_to_wav = 'path to training wav files' # needed to generate category dict

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

# Prediction example
train_manifest = Manifest(path_to_wav)
train_manifest.generate_category_dict()

predict_df = training_1k_df[training_1k_df.Talker == 'JUNIOR']

cosine_sims = Prediction(model, './earshot/checkpoints/cp-06000.ckpt', predict_df, training_1k_df)

cosine_sims.cosine_sim_dict.keys()
# plot individual word 
cosine_sims.plot_category_cosine('RIB', train_manifest.category_dict)
# plot grand mean
cosine_sims.plot_cosine_grand_mean(train_manifest.category_dict)
