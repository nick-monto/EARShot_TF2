import pandas as pd
import numpy as np
from earshot.data import pad
from earshot.model import EARSHOT
from earshot.parameters import ModelParameters

patt1 = pd.read_pickle(r'earshot/tests/test-input/FRUIT_AGNES.PICKLE')
patt2 = pd.read_pickle(r'earshot/tests/test-input/FRUIT_ALEX.PICKLE')
patt3 = pd.read_pickle(r'earshot/tests/test-input/FRUIT_ALLISON.PICKLE')
patt4 = pd.read_pickle(r'earshot/tests/test-input/FRUIT_AVA.PICKLE')
patt5 = pd.read_pickle(r'earshot/tests/test-input/FRUIT_BRUCE.PICKLE')

acoustics = [patt1['Acoustic'],
             patt2['Acoustic'],
             patt3['Acoustic'],
             patt4['Acoustic'],
             patt5['Acoustic']]

padded_acoustics = []
for i in acoustics:
    # add small noise so masking doesn't affect sound
    i += 1e-6
    padded_acoustics.append(pad(i, (54, 256)))

semantics = np.stack((patt1['Semantic'],
                      patt2['Semantic'],
                      patt3['Semantic'],
                      patt4['Semantic'],
                      patt5['Semantic']))

p = ModelParameters()

# model = Earshot(input_shape=(np.array(padded_acoustics).shape[1],
#                             np.array(padded_acoustics).shape[2]),
#                output_len=semantics.shape[1],
#                batch_size=1,p)

# model.fit(
#    x=np.array(padded_acoustics),
#    y=semantics,
#    batch_size=1
# )

# testing sub-classed model
model = EARSHOT(semantics.shape[1], p)
model.compile(loss=model.loss, optimizer="adam")
model.fit(x=np.array(padded_acoustics),
          y=semantics,
          batch_size=1
          )

# check dimensions of weights for each layer
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for name, weight in zip(names, weights):
    print(name, weight.shape)

# model summary
model.model((54, 256)).summary()
