import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Masking, LSTM

'''
We should eventually use a container class or derived-from-model class because of
the proliferation of input arguments (since we need the model construction parameters)
'''
def Earshot(input_shape, output_len, batch_size, model_parameters):
    inputs = Input(shape=input_shape, batch_size=batch_size, name="input")
    x = Masking(mask_value=0, name="mask")(inputs)

    # creates the hidden layer based on what's in the input parameters
    if model_parameters.hidden['type'] == "LSTM":
        x = LSTM(model_parameters.hidden['size'], return_sequences=True, stateful=False, name="LSTM")(x)
    elif model_parameters.hidden['type'] == "GRU":
        x = GRU(model_parameters.hidden['size'], return_sequences=True, name= "GRU")(x)

    # loss function and output activation are coupled, this sets them both
    if model_parameters.train_loss == 'CE':
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    elif model_parameters.train_loss == 'MSE':
        loss = "mean_squared_error"
        activation = tf.nn.tanh

    # create the output layer with the appropriate activation function; model
    #   is compiled with the corresponding matching loss function
    outputs = Dense(output_len, activation=activation, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)

    # create the optimizer

    model.compile(loss=loss, optimizer="adam")
    return model
