import tensorflow as tf
import pandas as pd

from earshot.audio import *
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

# check number of available physical gpus
print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

# instantiate model callbacks for early stopping and model checkpointing
es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=100) # stop early if no significant change in loss
filepath = './earshot/checkpoints/cp-{epoch:05d}.ckpt'
# save a model checkpoint after every 500 epochs
# TODO might change to every k number of samples
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='loss', 
                                                verbose=1, save_freq='epoch', period=500)

# Instantiate mirrored strategy for training locally on multiple gpus
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# generate model variable within the strategy scope
# sub-classed model requires length of target vector and model parameters
with strategy.scope():
    model = EARSHOT(training_1k['Target'][0].shape[0], p)
    model.compile(loss=model.loss, optimizer="adam", metrics=[
                  tf.keras.metrics.CategoricalAccuracy()])

model.fit(
    batch_gen,
    epochs=10000,
    callbacks=[checkpoint, es],
    shuffle=True
)

# # Prediction example
# # have to pass talker within a list ['JUNIOR'], but can assign multiple via ['ALLISON', 'JUNIOR', etc.]
# predict_df = train_manifest.gen_predict(['JUNIOR'], subset=subset)

# cosine_sims = Prediction(model, predict_df, training_1k)

# cosine_sims.cosine_sim_dict.keys()
# cosine_sims.plot_category_cosine('LARD', train_manifest.category_dict)
