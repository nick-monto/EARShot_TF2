from earshot.audio import *
import numpy as np
from scipy import spatial
import tensorflow as tf
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

# log desired audio files, provide path to folder containing them
train_manifest = Manifest('insert path to directory containing audio files')

# generate target SRVs
train_manifest.generate_srvs(target='Word', target_len=300, target_on=10)

# create batch generator of 100 items from the input manifest excluding the talker Junior
# remove the .sample() to create a batch generator of the whole set
# scipy spectrogram calcs are a lot slower, but may be due to being written as a class
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
    shuffle=True,
    use_multiprocessing=False
)

# Prediction example
test = spectro_calc(train_manifest.manifest['Path'].tolist()[0])
test_prediction = model.predict(test.reshape(1, -1, 256))

results = [1 - spatial.distance.cosine(train_manifest.manifest['Target'].tolist()[
                                       0], i) for i in test_prediction[0]]
np.max(results)
