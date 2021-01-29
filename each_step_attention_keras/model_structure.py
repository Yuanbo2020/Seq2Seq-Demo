import tensorflow as tf
import config as config
import tensorflow_addons as tfa
import numpy as np


def attention_3d_block(inputs, TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, lstm_units)

    # (batch_size, time_steps, lstm_units) -> (batch_size, lstm_units, time_steps)
    a = tf.keras.layers.Permute((2, 1))(inputs)

    # 对最后一维进行全连接
    # (batch_size, lstm_units, time_steps) -> (batch_size, lstm_units, time_steps)
    a = tf.keras.layers.Dense(TIME_STEPS, activation='softmax')(a)

    # (batch_size, lstm_units, time_steps) -> (batch_size, time_steps, lstm_units)
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)

    # 相乘
    # 相当于获得每一个step中，每个维度在所有step中的权重
    output_attention_mul = tf.keras.layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def get_each_step_attention_model(TIME_STEPS, INPUT_DIM):
    inputs = tf.keras.layers.Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    # (batch_size, time_steps, INPUT_DIM) -> (batch_size, input_dim, lstm_units)
    lstm_out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out, TIME_STEPS)
    # (batch_size, input_dim, lstm_units) -> (batch_size, input_dim*lstm_units)
    attention_mul = tf.keras.layers.Flatten()(attention_mul)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)
    model = tf.keras.Model(inputs=[inputs], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


#-------------------------------------------#
#   对每一个step的INPUT_DIM的attention几率
#   求平均
#-------------------------------------------#
def get_activations(model, inputs, layer_name=None):
    inp = model.input
    for layer in model.layers:
        if layer.name == layer_name:
            Y = layer.output
    model = tf.keras.Model(inp, Y)
    out = model.predict(inputs)
    out = np.mean(out[0],axis=-1)
    return out






