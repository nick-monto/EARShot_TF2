from numpy import power,min,max,nan,floor
import tensorflow as tf
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import Dense, GRU, Input, Masking, LSTM
from tensorflow.keras.callbacks import LearningRateScheduler

tf.keras.backend.set_floatx('float64')

'''
Learning rate adjustment functions.
'''
def noam_decay_lr(initial,warmup,minimum):
    '''
    Wrapper to define noam decay; the wrapper method allows us to make the
    lr update depend on additional parameters.

    The maximal learning rate under this scheme occurs at epoch = warmup, and
    will be equal to initial/warmup.
    '''
    def schedule(epoch):
        lr = initial*power(warmup,-0.5)*min([(epoch+1)*power(warmup,-1.5),power(epoch+1,-0.5)])

    return LearningRateScheduler(schedule)


def step_decay_lr(initial,drop_factor,drop_every):
    '''
    Wrapper that just drops the learning rate by a fixed factor (drop_factor) every drop_every
    epochs.
    '''
    def schedule(epoch):
        exp_fac = floor((1+epoch)/drop_every)
        lr = initial*power(drop_factor,exp_fac)
        return lr

    return LearningRateScheduler(schedule)


def polynomial_decay_lr(initial,max_epochs,poly_pow):
    '''
    Wrapper that drops the learning rate to zero over max_epochs epochs, with
    shape given by poly_pow (set poly_pow = 1 to get linear decay).
    '''
    def schedule(epoch):
        decay = power((1 - (epoch/max_epochs)),poly_pow)
        lr = initial*decay
        return lr

    return LearningRateScheduler(schedule)


def constant_lr(initial):
    '''
    Wrapper that just clamps the learning rate at the initial value forever.
    '''
    def schedule(epoch):
        return initial

    return LearningRateScheduler(schedule)



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
                             return_sequences=True, stateful=False,
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

        # set learning rate schedule
        if self.model_parameters.learning_schedule == 'noam':
            self.lr_sched = noam_decay_lr(**self.model_parameters['noam'])
            learning_rate = self.model_parameters['noam']['initial']
        elif self.model_parameters.learning_schedule == 'constant':
            self.lr_sched = constant_lr(self.model_parameters['constant']['rate'])
            learning_rate = self.model_parameters['constant']['rate']

        # optimizer
        if self.model_parameters.optimizer == 'ADAM':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,**tf.keras.optimizers['ADAM'])
        elif self.model_parameters.optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,**tf.keras.optimizers['SGD'])

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
