import tensorflow as tf
import numpy as np
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Prenet(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()

        for filters, kernel_size, stride in zip(
            hp_Dict['Model']['Prenet']['Filters'],
            hp_Dict['Model']['Prenet']['Kernel_Size'],
            hp_Dict['Model']['Prenet']['Strides']
            ):
            self.layer.add(tf.keras.layers.Conv1D(
                filters= filters,
                kernel_size= kernel_size,
                strides= stride,
                padding= 'same',
                use_bias= not hp_Dict['Model']['Prenet']['Use_Batch_Normalization']
                ))
            if hp_Dict['Model']['Prenet']['Use_Batch_Normalization']:
                self.layer.add(tf.keras.layers.BatchNormalization())
            self.layer.add(tf.keras.layers.ReLU())
            if hp_Dict['Model']['Prenet']['Dropout_Rate'] > 0.0:
                self.layer.add(tf.keras.layers.Dropout(
                    rate= hp_Dict['Model']['Prenet']['Dropout_Rate'])
                    )

        self.built = True

    def call(self, inputs):
        '''
        inputs: acoustic pattern    [Batch, Time, Dim]
        '''
        return self.layer(inputs)

class Network(tf.keras.layers.Layer):
    def build(self, input_shapes):
        if hp_Dict['Model']['Hidden']['Type'].upper() == 'LSTM':
            rnn_Cell = tf.keras.layers.LSTM
        elif hp_Dict['Model']['Hidden']['Type'].upper() == 'GRU':
            rnn_Cell = tf.keras.layers.GRU
        elif hp_Dict['Model']['Hidden']['Type'].upper() == 'BPTT':
            rnn_Cell = tf.keras.layers.SimpleRNN
        else:
            raise ValueError('Unsupported hidden rnn type.')
        
        self.layer_Dict = {}
        self.layer_Dict['Hidden'] = rnn_Cell(
            units= hp_Dict['Model']['Hidden']['Size'],
            return_sequences= True,
            return_state= True
            )
        self.layer_Dict['Output'] = tf.keras.layers.Dense(
            units= hp_Dict['Pattern']['Semantic'][hp_Dict['Pattern']['Semantic']['Mode'].upper()]['Size']
            )

        self.built = True

    def call(self, inputs):
        '''
        inputs: prenet(acoustic) and previous_state pattern  
        prenet(acoustic): [Batch, Time, Dim]
        previous_state:
        A list of [[Batch, Dim], [Batch, Dim]] when type is LSTM.
        A tensor of [Batch, Dim] when type is not LSTM.
        '''
        prenets, states = inputs
        states = tf.nest.map_structure(
            lambda x: tf.tile(x, [tf.shape(prenets)[0], 1]),
            states
            )

        new_Tensor = self.layer_Dict['Hidden'](prenets, initial_state= states)

        if hp_Dict['Model']['Hidden']['Type'].upper() == 'LSTM':
            new_Tensor, new_State = new_Tensor[0], [new_Tensor[1], new_Tensor[2]]
        elif hp_Dict['Model']['Hidden']['Type'].upper() in ['GRU', 'BPTT']:
            new_Tensor, new_State = new_Tensor
        else:
            raise ValueError('Unsupported hidden rnn type.')

        hidden_Tensor = new_Tensor
        new_Tensor = self.layer_Dict['Output'](new_Tensor)

        new_State = tf.nest.map_structure(
            lambda x: tf.expand_dims(x[0, :], axis= 0),
            new_State
            )

        return new_Tensor, hidden_Tensor, new_State

    def get_initial_state(self, **kwargs):
        initial_State = np.zeros((1, hp_Dict['Model']['Hidden']['Size']))

        if hp_Dict['Model']['Hidden']['Type'].upper() == 'LSTM':
            return [initial_State] * 2
        elif hp_Dict['Model']['Hidden']['Type'].upper() in ['GRU', 'BPTT']:
            return initial_State
        else:
            raise ValueError('Unsupported hidden rnn type.')

class Loss(tf.keras.layers.Layer):
    def call(self, inputs):
        '''
        inputs: acoustic lengths, semantic lables and logits
        lengths: [Batch,]
        labels: [Batch, Dim]
        logits: [Batch, Time, Dim]
        '''
        lengths, labels, logits = inputs

        labels = tf.tile(
            tf.expand_dims(labels, axis= 1),
            [1, tf.shape(logits)[1], 1]
            )
        loss = tf.nn.sigmoid_cross_entropy_with_logits(                     
            labels= tf.cast(labels, dtype= tf.float32),
            logits= tf.cast(logits, dtype= tf.float32)
            )
        loss *= tf.expand_dims(
            tf.sequence_mask(
                lengths= lengths,
                dtype= loss.dtype
                ),
            axis= -1
            )
        
        loss_Sequence = tf.reduce_mean(loss, axis= -1)
        loss_Sequence /= tf.math.count_nonzero(
            loss_Sequence,
            axis= 0,
            keepdims= True,
            dtype= loss_Sequence.dtype
            )
        loss_Sequence = tf.reduce_sum(loss_Sequence, axis= 0)

        loss = tf.reduce_sum(tf.reduce_mean(loss, axis= -1))

        return loss, loss_Sequence


class NoamDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        warmup_steps= 4000,
        min_learning_rate= None,
        ):    
        super(NoamDecay, self).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.min_learning_rate = min_learning_rate or 0

    def __call__(self, step):
        learning_rate = self.initial_learning_rate * self.warmup_steps ** 0.5 * tf.minimum(step * self.warmup_steps**-1.5, step**-0.5)        
        return tf.maximum(learning_rate, self.min_learning_rate)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'min_learning_rate': self.min_learning_rate
            }

