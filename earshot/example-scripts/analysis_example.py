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

# Prediction example
train_manifest = Manifest(path_to_wav)
train_manifest.generate_category_dict()

predict_df = training_1k_df[training_1k_df['Talker'].isin(['JUNIOR','AGNES'])]

cosine_sims = Prediction(model, './earshot/checkpoints/cp-06000.ckpt', predict_df, training_1k_df)

# plot category flow of an individual word 
cosine_sims.plot_category_cosine('RIB_JUNIOR', train_manifest.category_dict)
# plot category flow of the grand mean
cosine_sims.plot_cosine_grand_mean(train_manifest.category_dict)

# Calculate accuracy/rt
accuracy = []
e = 1000
while e <= 6000:
    cosine_sims = Prediction(model, './earshot/checkpoints/cp-0'+str(e)+'.ckpt', predict_df, training_1k_df)
    rt_acc_list = []
    for word in tqdm(list(cosine_sims.cosine_sim_dict.keys())):
        keys, values = list(cosine_sims.gen_RT(word, predict_df).keys()), list(cosine_sims.gen_RT(word, predict_df).values())
        rt_acc_list.append((word, *values))

    rt_acc_df = pd.DataFrame(rt_acc_list, columns=['Word', 'Absolute', 'Relative', 'Time Dependent'])

    accuracy.append((e, 1 - rt_acc_df.isna().sum()['Absolute'] / len(rt_acc_df),
                    1 - rt_acc_df.isna().sum()['Relative'] / len(rt_acc_df),
                    1 - rt_acc_df.isna().sum()['Time Dependent'] / len(rt_acc_df)))
    e += 1000

accuracy_df = pd.DataFrame(accuracy, columns=['Epoch','Absolute','Relative','Time Dependent'])

accuracy_df.set_index('Epoch').plot.line(xlabel='Epoch',ylabel='Accuracy', ylim=(0,1), title="Threshold = 0.40; Junior tested")
plt.show()
