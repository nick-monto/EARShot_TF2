import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import json, os, time, argparse, pickle
from threading import Thread
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import wavfile

from Feeder import Feeder
import Modules
from ProgressBar import progress


with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

if not hp_Dict['Device'] is None:
    os.environ["CUDA_VISIBLE_DEVICES"]= hp_Dict['Device']

if hp_Dict['Use_Mixed_Precision']:    
    policy = mixed_precision.Policy('mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

class EARShot:
    def __init__(self, start_Epoch= 0, is_Training= False):
        self.feeder = Feeder(
            start_Epoch= start_Epoch,
            is_Training= is_Training
            )
        if start_Epoch == 0:
            self.Export_Training_Metadata()

        self.Model_Generate()

    def Model_Generate(self):
        input_Dict = {}
        layer_Dict = {}
        tensor_Dict = {}

        if hp_Dict['Pattern']['Acoustic']['Mode'].upper() == 'Spectrogram'.upper():
            acoustic_Size = hp_Dict['Pattern']['Acoustic']['Spectrogram']['Dimension']
        if hp_Dict['Pattern']['Acoustic']['Mode'].upper() == 'Mel'.upper():
            acoustic_Size = hp_Dict['Pattern']['Acoustic']['Mel']['Dimension']
        input_Dict['Acoustic'] = tf.keras.layers.Input(
            shape= [None, acoustic_Size],
            dtype= tf.as_dtype(policy.compute_dtype)
            )
        input_Dict['Acoustic_Length'] = tf.keras.layers.Input(
            shape= [],
            dtype= tf.as_dtype(tf.int32)
            )
        
        if hp_Dict['Model']['Hidden']['Type'].upper() == 'LSTM':
            input_Dict['Previous_State'] = [
                tf.keras.layers.Input(
                    shape= [hp_Dict['Model']['Hidden']['Size'],],
                    dtype= tf.as_dtype(policy.compute_dtype)
                    ),        
                tf.keras.layers.Input(
                    shape= [hp_Dict['Model']['Hidden']['Size'],],
                    dtype= tf.as_dtype(policy.compute_dtype)
                    )
                ]
        elif hp_Dict['Model']['Hidden']['Type'].upper() in ['GRU', 'BPTT']:
            input_Dict['Previous_State'] = tf.keras.layers.Input(
                shape= [hp_Dict['Model']['Hidden']['Size'],],
                dtype= tf.as_dtype(policy.compute_dtype)
                )

        if hp_Dict['Pattern']['Semantic']['Mode'].upper() == 'SRV':
            semantic_Size = hp_Dict['Pattern']['Semantic']['SRV']['Size']
        if hp_Dict['Pattern']['Semantic']['Mode'].upper() == 'PGD':
            semantic_Size = hp_Dict['Pattern']['Semantic']['PGD']['Size']
        input_Dict['Semantic'] = tf.keras.layers.Input(
            shape= [semantic_Size,],
            dtype= tf.as_dtype(policy.compute_dtype)
            )

        layer_Dict['Prenet'] = Modules.Prenet()
        layer_Dict['Network'] = Modules.Network()
        layer_Dict['Loss'] = Modules.Loss()

        
        if hp_Dict['Model']['Prenet']['Use']:
            tensor_Dict['Prenet'] = layer_Dict['Prenet'](input_Dict['Acoustic'])
        else:
            tensor_Dict['Prenet'] = input_Dict['Acoustic']

        tensor_Dict['Semantic'], tensor_Dict['Hidden'], tensor_Dict['State'] = layer_Dict['Network']([
            tensor_Dict['Prenet'],
            input_Dict['Previous_State']
            ])
        tensor_Dict['Loss'], layer_Dict['Loss_Sequence'] = layer_Dict['Loss']([
            input_Dict['Acoustic_Length'],
            input_Dict['Semantic'],
            tensor_Dict['Semantic']
            ])

        self.model_Dict = {}
        self.model_Dict['Train'] = tf.keras.Model(
            inputs= [
                input_Dict['Acoustic'],
                input_Dict['Acoustic_Length'],
                input_Dict['Previous_State'],
                input_Dict['Semantic']
                ],
            outputs= [
                tensor_Dict['Loss'],
                layer_Dict['Loss_Sequence'],
                tensor_Dict['State']
                ]
            )
        self.model_Dict['Test'] = tf.keras.Model(
            inputs= [
                input_Dict['Acoustic'],
                input_Dict['Previous_State']
                ],
            outputs= tensor_Dict['Semantic']
            )
        self.model_Dict['Hidden'] = tf.keras.Model(
            inputs= [
                input_Dict['Acoustic'],
                input_Dict['Previous_State']
                ],
            outputs= tensor_Dict['Hidden']
            )
        
        self.model_Dict['Train'].summary()
        
        if hp_Dict['Train']['Learning_Rate']['Use_Noam']:
            learning_Rate = Modules.NoamDecay(
                initial_learning_rate= hp_Dict['Train']['Learning_Rate']['Initial'],
                warmup_steps= hp_Dict['Train']['Learning_Rate']['Warmup_Step'],
                min_learning_rate= hp_Dict['Train']['Learning_Rate']['Min']
                )
        else:
            learning_Rate = hp_Dict['Train']['Learning_Rate']['Initial']

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= learning_Rate,
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            clipnorm= 1.0
            )
        self.checkpoint = tf.train.Checkpoint(optimizer= self.optimizer, model= self.model_Dict['Train'])

        self.get_Initial_Hidden_State = layer_Dict['Network'].get_initial_state

    def Train_Step(self, acoustics, acoustic_Steps, states, semantics):
        with tf.GradientTape() as tape:
            loss, loss_Sequence, state = self.model_Dict['Train'](
                inputs= [acoustics, acoustic_Steps, states, semantics],
                training= True
                )
        gradients = tape.gradient(
            loss,
            self.model_Dict['Train'].trainable_variables
            )
        self.optimizer.apply_gradients([
            (gradient, variable)            
            for gradient, variable in zip(gradients, self.model_Dict['Train'].trainable_variables)
            ])

        return loss, loss_Sequence, state

    def Test_Step(self, acoustics, states):
        logits = self.model_Dict['Test'](
            inputs= [acoustics, states],
            training= False
            )

        return logits

    def Hidden_Step(self, acoustics):
        hiddens = self.model_Dict['Hidden'](
            inputs= [acoustics],
            training= False
            )

        return hiddens

    def Restore(self, checkpoint_File_Path= None):
        if checkpoint_File_Path is None:
            checkpoint_File_Path = tf.train.latest_checkpoint(os.path.join(hp_Dict['Result_Path'], 'Checkpoint'))

        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.checkpoint.restore(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        if not self.feeder.is_Training:
            print('Model is not run in the training mode.')
            return

        def Save_Checkpoint(epoch):
            os.makedirs(os.path.join(hp_Dict['Result_Path'], 'Checkpoint').replace("\\", "/"), exist_ok= True)
            self.checkpoint.save(
                os.path.join(hp_Dict['Result_Path'], 'Checkpoint', 'E_{}.CHECKPOINT.H5'.format(epoch)).replace('\\', '/')
                )
           
        def Run_Test(epoch):
            return self.Test(epoch)

        states = self.get_Initial_Hidden_State()
        while not self.feeder.is_Finished or len(self.feeder.pattern_Queue) > 0:    #When there is no more training pattern, the train function will be done.
            epoch, is_New_Epoch, pattern_Dict = self.feeder.Get_Pattern()

            if is_New_Epoch:
                if epoch % (hp_Dict['Train']['Checkpoint_Save_Timing']) == 0:
                    Save_Checkpoint(epoch)
                if epoch % (hp_Dict['Train']['Test_Timing']) == 0:
                    Run_Test(epoch)

            start_Time = time.time()
            loss, loss_Sequence, states = self.Train_Step(**{
                **pattern_Dict,
                'states': states if hp_Dict['Model']['Hidden']['No_Reset_State'] else self.get_Initial_Hidden_State()
                })

            if np.isnan(loss) or np.isinf(np.abs(loss)):
                raise Exception('Because NaN/Inf loss is generated.')

            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(self.optimizer.iterations.numpy()),
                'Epoch: {}'.format(epoch),
                'LR: {:0.7f}'.format(
                    self.optimizer.lr(self.optimizer.iterations.numpy() - 1)
                    if hp_Dict['Train']['Learning_Rate']['Use_Noam'] else
                    hp_Dict['Train']['Learning_Rate']['Initial']
                    ),
                'Loss.M: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))
            print('\t'.join(['{:0.5f}'.format(x) for x in loss_Sequence.numpy()]))

            with open(os.path.join(hp_Dict['Result_Path'], 'Test', 'log.txt').replace("\\", "/"), 'a') as f:
                f.write('\t'.join([
                '{:0.3f}'.format(time.time() - start_Time),
                '{}'.format(self.optimizer.iterations.numpy()),
                '{}'.format(epoch),
                '{:0.7f}'.format(
                    self.optimizer.lr(self.optimizer.iterations.numpy() - 1)
                    if hp_Dict['Train']['Learning_Rate']['Use_Noam'] else
                    hp_Dict['Train']['Learning_Rate']['Initial']
                    ),
                '{:0.5f}'.format(loss)
                ]) + '\n')

        Save_Checkpoint(epoch + 1)
        Run_Test(epoch + 1).join()  # Wait until finishing the test and extract the data.

    def Test(self, epoch):
        infos, pattern_List = self.feeder.Get_Test_Pattern()
        
        logits = []
        for batch_Index, patterns in enumerate(pattern_List):
            logits.append(self.Test_Step(**{**patterns, 'states': self.get_Initial_Hidden_State()}).numpy())
            progress(
                batch_Index + 1,
                len(pattern_List),
                status='Testing'
                )
        print()
        logits = np.vstack(logits)

        export_Thread = Thread(
            target=self.Export_Test,
            args=(infos, logits, epoch)
            )  #Turning on a new thread for saving result
        export_Thread.daemon = True
        export_Thread.start()

        return export_Thread

    def Export_Test(self, infos, logits, epoch):
        os.makedirs(os.path.join(hp_Dict['Result_Path'], 'Test').replace("\\", "/"), exist_ok= True)

        for start_Index in range(0, len(infos), hp_Dict['Train']['Batch_Size']):
            result_Dict = {}
            result_Dict["Epoch"] = epoch
            result_Dict["Start_Index"] = start_Index
            result_Dict["Info"] = infos[start_Index:start_Index + hp_Dict['Train']['Batch_Size']]
            result_Dict["Result"] = logits[start_Index:start_Index + hp_Dict['Train']['Batch_Size']]
            result_Dict["Exclusion_Ignoring"] = \
                epoch > hp_Dict['Train']['Max_Epoch_with_Exclusion'] and \
                epoch <= hp_Dict['Train']['Max_Epoch_without_Exclusion']    #The results need to be judged if they are contaminated with excluded pattern. Determine if the model has been exposed to the excluded pattern.
                
            with open(os.path.join(hp_Dict['Result_Path'], 'Test', 'E_{:06d}.I_{:09d}.pickle'.format(epoch, start_Index)).replace("\\", "/"), "wb") as f:
                pickle.dump(result_Dict, f, protocol=4)

    def Export_Training_Metadata(self):
        os.makedirs(hp_Dict['Result_Path'], exist_ok= True)

        with open(os.path.join(hp_Dict['Result_Path'], 'Hyper_Parameters.json').replace("\\", "/"), "w") as f:
                json.dump(hp_Dict, f, indent= 4)

        with open(os.path.join(hp_Dict['Result_Path'], 'Training_Metadta.pickle').replace("\\", "/"), "wb") as f:
            pickle.dump(self.feeder.pattern_Path_Dict, f, protocol=4)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--epoch", default= 0, type= int)
    argument_Dict = vars(argParser.parse_args())

    new_Model = EARShot(is_Training= True, start_Epoch= argument_Dict['epoch'])
    new_Model.Restore()
    new_Model.Train()
    