from data_generator import DateData
import sys
from model_structure import *


def main(argv):
    # get and process data
    data = DateData(config.num_samples)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    training = 1
    inference = 1

    bx, by, decoder_len, data_cn, data_en = data.sample(config.batch_size)

    model, basicdecoder, decoder_cell, decoder_dense, \
    enc_embeddings_layer, dec_embeddings_layer, en_lstm_layer = seq2seq_model(bx, decoder_len, data.num_word,
                                                                              data.num_word)

    if training:
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        for step in range(config.epochs):
            bx, by, decoder_len, data_cn, data_en = data.sample(config.batch_size)
            with tf.GradientTape() as tape:
                enc_embeddings = enc_embeddings_layer(bx)
                o, h, c = en_lstm_layer(enc_embeddings)
                s = [h, c]
                dec_in = by[:, :-1]  # ignore <EOS>
                dec_emb_in = dec_embeddings_layer(dec_in)
                o, hyb1, hyb2 = basicdecoder(dec_emb_in, s, sequence_length=decoder_len)
                logits = o.rnn_output
                dec_out = by[:, 1:]  # ignore <GO>
                loss = cross_entropy(dec_out, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 70 == 0:
                enc_embeddings = enc_embeddings_layer(bx[0:1])
                o, h, c = en_lstm_layer(enc_embeddings)
                s = [h, c]
                target = data.idx2str(by[0, 1:-1])
                pred = inference_sequence(bx[0:1], s, decoder_cell,
                                     decoder_dense, dec_embeddings_layer,
                                     data.start_token, data.end_token, max_pred_len=11)
                res = data.idx2str(pred[0])
                src = data.idx2str(bx[0])
                print(
                    "validation step: ", step,
                    "| loss: %.3f" % loss,
                    "| input: ", src,
                    "| target: ", target,
                    "| inference: ", res,
                )
        model.save_weights(config.model_path)

    if inference:
        model.load_weights(config.model_path)
        bx, by, decoder_len, data_cn, data_en = data.sample(config.batch_size)
        for i in range(config.batch_size):
            sub_x = np.expand_dims(bx[i], axis=0)
            sub_y = np.expand_dims(by[i], axis=0)
            enc_embeddings = enc_embeddings_layer(sub_x)
            o, h, c = en_lstm_layer(enc_embeddings)
            s = [h, c]
            target = data.idx2str(sub_y[0, 1:-1])
            pred = inference_sequence(sub_x, s, decoder_cell,
                                 decoder_dense, dec_embeddings_layer,
                                 data.start_token, data.end_token, by.shape[1])
            res = data.idx2str(pred[0])
            src = data.idx2str(sub_x[0])
            print(
                "Testing sample: ", i,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


