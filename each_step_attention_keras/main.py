import sys
from data_generator import get_data_recurrent
from model_structure import *
import matplotlib.pyplot as plt
import pandas as pd


def main(argv):
    X, Y = get_data_recurrent(config.N, config.TIME_STEPS, config.INPUT_DIM)

    model = get_each_step_attention_model(config.TIME_STEPS, config.INPUT_DIM)

    model.fit(X, Y, epochs=config.epochs, batch_size=config.batch_size,
              validation_split=config.validation_split)

    attention_vectors = []
    for i in range(300):
        testing_X, testing_Y = get_data_recurrent(1, config.TIME_STEPS, config.INPUT_DIM)
        attention_vector = get_activations(model, testing_X, layer_name='attention_vec')
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


