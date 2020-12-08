import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
from earshot.audio import *
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

# log desired audio files, provide path to folder containing them
train_manifest = Manifest('path to input directory')

# generate target SRVs
train_manifest.generate_srvs(target='Word', target_len=300, target_on=10)

# generate category dictionary
train_manifest.generate_category_dict()

# select the subset of ~1000 words
subset = pd.read_csv('./Pronunciation_Data_1K.New.txt', sep='\t', names=['Word','Pronounce'])
training_1k = train_manifest.manifest[train_manifest.manifest['Word'].isin(subset['Word'].tolist())]

# create batch generator of 100 items from the input manifest excluding the talker Junior
# remove the .sample() to create a batch generator of the whole set
batch_gen = DataGenerator(df=training_1k[training_1k.Talker != 'JUNIOR'].sample(100),
                          batch_size=16, pad_value=-9999, return_seq=True, spec_calc='scipy')

p = ModelParameters()

# instantiate model
# sub-classed model requires length of target vector and model parameters
model = EARSHOT(training_1k['Target'][0].shape[0], p)
model.compile(loss=model.loss, optimizer="adam", metrics=[
              tf.keras.metrics.CategoricalAccuracy()])
model.fit(
    batch_gen,
    epochs=1,
    shuffle=True
)

# Prediction example
# have to pass talker within a list ['JUNIOR'], but can assign multiple via ['ALLISON', 'JUNIOR', etc.]
predict_df = train_manifest.gen_predict(['JUNIOR'], subset=subset)

cosine_sims = Prediction(model, predict_df, training_1k)

cosine_sims.cosine_sim_dict.keys()
cosine_sims.plot_category_cosine('LARD', train_manifest.category_dict)
