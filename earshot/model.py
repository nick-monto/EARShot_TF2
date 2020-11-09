import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, GRU, Input, Masking, LSTM

tf.keras.backend.set_floatx('float64')

'''
We should eventually use a container class or derived-from-model class because of
the proliferation of input arguments (since we need the model construction parameters)
'''

# def Earshot(input_shape, output_len, batch_size, model_parameters):
#     inputs = Input(shape=input_shape, batch_size=batch_size, name="input")
#     x = Masking(mask_value=0, name="mask")(inputs)

#     # creates the hidden layer based on what's in the input parameters
#     if model_parameters.hidden['type'] == "LSTM":
#         x = LSTM(model_parameters.hidden['size'],
#                  return_sequences=True, stateful=False, name="LSTM")(x)
#     elif model_parameters.hidden['type'] == "GRU":
#         x = GRU(model_parameters.hidden['size'],
#                 return_sequences=True, name="GRU")(x)

#     # loss function and output activation are coupled, this sets them both
#     if model_parameters.train_loss == 'CE':
#         loss = "binary_crossentropy"
#         activation = tf.nn.sigmoid
#     elif model_parameters.train_loss == 'MSE':
#         loss = "mean_squared_error"
#         activation = tf.nn.tanh

#     # create the output layer with the appropriate activation function; model
#     #   is compiled with the corresponding matching loss function
#     outputs = Dense(output_len, activation=activation, name="output")(x)
#     model = Model(inputs=inputs, outputs=outputs)

#     # create the optimizer

#     model.compile(loss=loss, optimizer="adam")
#     return model

# sub-classing from keras Model

class EARSHOT(Model):
    '''
    EARSHOT model sub-classing tf.keras.Model
    '''
    def __init__(self, output_len, model_parameters):
        '''
        output_len = length of target vector
        model_parameters = model hyper parameters pulled from parameters.py
        '''
        super(EARSHOT, self).__init__(name='earshot')
        self.model_parameters = model_parameters
        self.mask = Masking(mask_value=-9999, name="mask")

        if self.model_parameters.hidden['type'] == "LSTM":
            self.hidden = LSTM(self.model_parameters.hidden['size'],
                             return_sequences=False, stateful=False,
                             name="LSTM")
        elif self.model_parameters.hidden['type'] == "GRU":
            self.hidden = GRU(self.model_parameters.hidden['size'],
                           return_sequences=True, name="GRU")

        # loss function and output activation are coupled, this sets them both
        if self.model_parameters.train_loss == 'CE':
            self.loss = "binary_crossentropy"
            self.activation = tf.nn.sigmoid
        elif self.model_parameters.train_loss == 'MSE':
            self.loss = "mean_squared_error"
            self.activation = tf.nn.tanh

        self.dense_output = Dense(output_len, activation=self.activation)


    def call(self, inputs):
        '''
        Input is provided at training time.
        '''
        x = self.mask(inputs)
        x = self.hidden(x)
        return self.dense_output(x)

    def model(self, input_shape):
        '''
        Function for model introspection
        '''
        x = Input(input_shape)
        return Model(inputs=[x], outputs=self.call(x))
