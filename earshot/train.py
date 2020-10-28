import pandas as pd
import numpy as np
from earshot.model import Earshot

object = pd.read_pickle(r'./earshot/test-input/FRUIT_AGNES.PICKLE')
object2 = pd.read_pickle(r'./earshot/test-input/FRUIT_ALEX.PICKLE')
object3 = pd.read_pickle(r'./earshot/test-input/FRUIT_ALLISON.PICKLE')
object4 = pd.read_pickle(r'./earshot/test-input/FRUIT_AVA.PICKLE')
object5 = pd.read_pickle(r'./earshot/test-input/FRUIT_BRUCE.PICKLE')

acoustics = [object['Acoustic'],
             object2['Acoustic'],
             object3['Acoustic'],
             object4['Acoustic'],
             object5['Acoustic']]

padded_acoustics = []
for i in acoustics:
    # add small noise so masking doesn't affect sound
    i += 1e-6
    if i.shape[0] < 54:
        while i.shape[0] < 54:
            i = np.vstack((i, np.zeros((256,))))
        padded_acoustics.append(i)
    else:
        padded_acoustics.append(i)

semantics = np.stack((object['Semantic'],
                      object2['Semantic'],
                      object3['Semantic'],
                      object4['Semantic'],
                      object5['Semantic']))

model = Earshot(input_shape=(np.array(padded_acoustics).shape[1],
                             np.array(padded_acoustics).shape[2]),
                output_len=semantics.shape[1],
                batch_size=1)

model.fit(
    x=np.array(padded_acoustics),
    y=semantics,
    batch_size=1
)
