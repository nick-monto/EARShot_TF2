import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Masking, LSTM


def Earshot(input_shape, output_len, batch_size):
    inputs = Input(shape=input_shape, batch_size=batch_size, name="input")
    x = Masking(mask_value=0, name="mask")(inputs)
    x = LSTM(512, return_sequences=True, stateful=False, name="LSTM")(x)
    outputs = Dense(output_len, activation=tf.nn.sigmoid, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
