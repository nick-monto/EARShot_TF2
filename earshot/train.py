from earshot.data import *
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

# log desired audio files, provide path to folder containing them
train_manifest = Manifest('include training directory here')

# generate target SRVs 
train_manifest.generate_labels(target='Word', target_len=300, target_on=10)

# create batch generator from the input manifest
# remove the .sample() to create a batch generator of the whole set
batch_gen = DataGenerator(df=train_manifest.manifest.sample(25), 
                          batch_size=5, pad_value=-9999)

p = ModelParameters()
# instantiate model
# sub-classed model requires length of target vector and model parameters
model = EARSHOT(train_manifest.manifest['Target'][0].shape[0], p)
model.compile(loss=model.loss, optimizer="adam")
model.fit(
    batch_gen,
    epochs = 5,
    shuffle = True,
    use_multiprocessing = False
)
