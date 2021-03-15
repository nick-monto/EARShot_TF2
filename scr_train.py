import tensorflow as tf
import pandas as pd

from earshot.audio import *
from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
# check number of available physical gpus
print("Number of GPUs Available: ", num_gpus)

# batches per gpu
batches = 2048
if num_gpus > 0:
    total_batch = batches*num_gpus
else:
    total_batch = batches*1

# log desired audio files, provide path to folder containing them
# NB: this path should be set in one of the 'parameters' objects
train_manifest = Manifest('/Users/kevinbrown/data/language/speech/WAV/EARSHOT_NEW_WAVS')

# generate target SRVs
# NB: this should be set in one of the 'parameters' objects
train_manifest.generate_srvs(target='Word', target_len=300, target_on=30)

# select the subset of ~1000 words
# NB: this should be set in one of the parameters objects
subset = pd.read_csv('Pronunciation_Data_1K.Modified.txt', sep='\t', names=['Word','Pronounce'])

small_subset = subset.sample(n=100)

# create training data
training_1k_df = train_manifest.calc_spec(subset=small_subset) # pre-compute spectrograms
training_1k_df.to_pickle('./earshot/tests/training-test.plk')

'''
try:
    training_1k_df = pd.read_pickle('./earshot/tests/training-test.plk')
except:
    print('No training pickle found, computing set.')
    training_1k_df = train_manifest.calc_spec(subset=subset) # pre-compute spectrograms
    training_1k_df.to_pickle('./earshot/tests/training-test.plk')
# generate category dictionary, only needed for analysis
# train_manifest.generate_category_dict()
'''

p = ModelParameters()

# instantiate model callbacks for early stopping and model checkpointing
# es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.003, patience=250) # stop early if no significant change in loss
# NB: this is probably something that should come from the parameters object
filepath = './earshot/checkpoints/cp-{epoch:05d}.ckpt'

# save a model checkpoint after every 250 epochs
# TODO might change to every k number of samples
# NB: save frequency should be set in a parameter (along with checkpoint location)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='loss',
                                                verbose=1, save_freq='epoch', period=250)
# log model training history
csv_logger = tf.keras.callbacks.CSVLogger('./earshot/model_history_log.csv', append=True)

if num_gpus > 1:
    # Instantiate mirrored strategy for training locally on multiple gpus
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # generate model variable within the strategy scope
    # sub-classed model requires length of target vector and model parameters
    with strategy.scope():
        model = EARSHOT(len(training_1k_df['Target'].iloc()[0]), p)
        model.compile(loss=model.loss, optimizer=model.optimizer, run_eagerly=False)
else:
        model = EARSHOT(len(training_1k_df['Target'].iloc()[0]), p)
        model.compile(loss=model.loss, optimizer=model.optimizer, run_eagerly=False)

# NB: number of training epochs should also be set in a parameters structure
model.fit(
    x = np.array(training_1k_df['Padded Input'].tolist()),
    y = np.array(training_1k_df['Padded Target'].tolist()),
    batch_size = total_batch,
    epochs=5,
    callbacks=[checkpoint, csv_logger, model.lr_sched],
    shuffle=True
)

# # Prediction example
# # have to pass talker within a list ['JUNIOR'], but can assign multiple via ['ALLISON', 'JUNIOR', etc.]
# predict_df = train_manifest.gen_predict(['JUNIOR'], subset=subset)

# cosine_sims = Prediction(model, predict_df, training_1k)

# cosine_sims.cosine_sim_dict.keys()
# cosine_sims.plot_category_cosine('LARD', train_manifest.category_dict)
