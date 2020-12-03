from earshot.audio import *
import numpy as np
from scipy import spatial
import tensorflow as tf
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

# log desired audio files, provide path to folder containing them
train_manifest = Manifest('path to input directory')

# generate target SRVs
train_manifest.generate_srvs(target='Word', target_len=300, target_on=10)

# create batch generator of 100 items from the input manifest excluding the talker Junior
# remove the .sample() to create a batch generator of the whole set
# scipy spectrogram calcs are a lot slower, likely due to calculating more time steps over a larger frequency range
batch_gen = DataGenerator(df=train_manifest.manifest[train_manifest.manifest.Talker != 'JUNIOR'].sample(100),
                          batch_size=16, pad_value=-9999, return_seq=True, spec_calc='scipy')

p = ModelParameters()

# instantiate model
# sub-classed model requires length of target vector and model parameters
model = EARSHOT(train_manifest.manifest['Target'][0].shape[0], p)
model.compile(loss=model.loss, optimizer="adam", metrics=[
              tf.keras.metrics.CategoricalAccuracy()])
model.fit(
    batch_gen,
    epochs=5,
    shuffle=True
)

# Prediction example
predict_df = train_manifest.gen_predict('JUNIOR')

cosine_sims = Prediction(model, predict_df, train_manifest.manifest)
cosine_sims.cosine_sim_dict
