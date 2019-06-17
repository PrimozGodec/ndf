import pickle
import os

from ndf.layers import Conv1D, Linear, TanH, Dense, Input, BatchNorm, \
    Flatten
from ndf.model import Model

# location of this file
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def encoder():
    # load weights
    with open(os.path.join(
            __location__, "weights", "encoder_weights.pkl"), "rb") as f:
        weights = pickle.load(f)

    input = Input(shape=(120, 35))

    x = Conv1D(9, 9, name="conv1d-1", kernel_weights=weights[0], bias_weights=weights[1])(input)
    x = TanH(name="tanh1")(x)
    x = BatchNorm(name="batchnorm1", gamma=weights[2], beta=weights[3], moving_mu=weights[4], moving_var=weights[5])(x)

    x = Conv1D(9, 9, name="conv1d-2", kernel_weights=weights[6], bias_weights=weights[7])(x)
    x = TanH(name="tanh2")(x)
    x = BatchNorm(name="batchnorm2", gamma=weights[8], beta=weights[9], moving_mu=weights[10], moving_var=weights[11])(
        x)

    x = Conv1D(10, 11, name="conv1d-3", kernel_weights=weights[12], bias_weights=weights[13])(x)
    x = TanH(name="tanh3")(x)
    x = BatchNorm(name="batchnorm3", gamma=weights[14], beta=weights[15], moving_mu=weights[16],
                  moving_var=weights[17])(x)

    x = Flatten(name="flatten")(x)

    x = Dense(name="dense1", kernel_weights=weights[18], bias_weights=weights[19])(x)
    x = TanH(name="tanh4")(x)

    x = BatchNorm(name="batchnorm4", gamma=weights[20], beta=weights[21], moving_mu=weights[22],
                  moving_var=weights[23])(x)

    z_mean = Dense(name="z_mean", kernel_weights=weights[24], bias_weights=weights[25])(x)
    z_mean = Linear(name="linear")(z_mean)

    model = Model([input], [z_mean])
    return model


if __name__ == '__main__':
    encoder()
