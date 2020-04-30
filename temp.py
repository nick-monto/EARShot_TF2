import tensorflow as tf

from Feeder import Feeder
import Modules

feeder = Feeder(0, True)

acoustics = tf.keras.layers.Input(
    shape= [None, 256],
    dtype= tf.float32
    )

net = Modules.Network()
lo = Modules.Loss()
logits, _, _ = net([acoustics, net.get_initial_state()])
model = tf.keras.Model(inputs= acoustics, outputs= logits)

patterns = feeder.Get_Pattern()[2]

optimizer = tf.keras.optimizers.Adam(
    learning_rate= 0.001,
    beta_1= 0.9,
    beta_2= 0.999,
    epsilon= 1e-7
    )

while True:
    with tf.GradientTape() as tape:
        logit = model(patterns['acoustics'])
        label = tf.expand_dims(patterns['semantics'], axis = 1)
        label = tf.tile(label, [1, tf.shape(logit)[1], 1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(label, logit)
        loss = tf.reduce_mean(loss)
        print(loss)

    gradients = tape.gradient(
        loss,
        model.trainable_variables
        )

    for gradient, variable in zip(gradients, model.trainable_variables):
        print(variable.name, '\t', tf.reduce_mean(tf.abs(gradient)))

    optimizer.apply_gradients([
        (gradient, variable)            
        for gradient, variable in zip(gradients, model.trainable_variables)
        ])