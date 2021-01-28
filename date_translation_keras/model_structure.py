import tensorflow as tf
import config as config
import tensorflow_addons as tfa
import numpy as np


def seq2seq_model(x, seq_len, enc_v_dim, dec_v_dim,
                  embedding_dim=config.embedding_dim, lstm_units=config.lstm_units):
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name='encoder_in')
    enc_embeddings_layer = tf.keras.layers.Embedding(input_dim=enc_v_dim, output_dim=embedding_dim,
                                               embeddings_initializer=tf.keras.initializers.RandomNormal(0., 0.1),
                                               name='encoder_emb')
    enc_embeddings = enc_embeddings_layer(encoder_inputs)

    init_s = [tf.zeros((x.shape[0], lstm_units)), tf.zeros((x.shape[0], lstm_units))]

    en_lstm_layer = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True, return_state=True,
                                   name='encoder_lstm')
    o, h, c = en_lstm_layer(enc_embeddings, initial_state=init_s)
    s = [h, c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name='decoder_in')
    dec_embeddings_layer = tf.keras.layers.Embedding(input_dim=dec_v_dim, output_dim=embedding_dim,  # [dec_n_vocab, emb_dim]
                                               embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
                                               name='decoder_emb')
    dec_embeddings = dec_embeddings_layer(decoder_inputs)

    # dec_embeddings shape: (32, 10, 16)
    decoder_cell = tf.keras.layers.LSTMCell(units=lstm_units, name='decoder_lstm')
    decoder_dense = tf.keras.layers.Dense(dec_v_dim, name='decoder_dense')
    basicdecoder = tfa.seq2seq.BasicDecoder(cell=decoder_cell,
                                             sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
                                             output_layer=decoder_dense,
                                             name='basic_decoder')

    o, hyb1, hyb2 = basicdecoder(dec_embeddings, s, sequence_length=seq_len)
    logits = o.rnn_output

    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                           outputs=logits,
                           name='seq2seq_model_from_Yuanbo')

    model.summary()

    return model, basicdecoder, decoder_cell, decoder_dense, \
           enc_embeddings_layer, dec_embeddings_layer, en_lstm_layer


def inference_sequence(x, s, decoder_cell, decoder_dense,
                       dec_embeddings, start_token,
                       end_token, max_pred_len):

        decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense)

        done, i, s = decoder_eval.initialize(
            dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], start_token),
            end_token=end_token,
            initial_state=s)

        pred_id = np.zeros((x.shape[0], max_pred_len), dtype=np.int32)
        for l in range(max_pred_len):
            o, s, i, done = decoder_eval.step(time=l, inputs=i, state=s, training=False)
            # `(outputs, next_state, next_inputs, finished)`.
            pred_id[:, l] = o.sample_id
        return pred_id