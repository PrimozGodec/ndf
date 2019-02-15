import numpy as np
from ndf.layers import Layer


class ReLU(Layer):

    number_of_inputs = 1

    def forward(self, x):
        return x * (x > 0)


class Softmax(Layer):

    number_of_inputs = 1

    def forward(self, x):
        exponents = np.exp(x)
        row_sums = np.sum(exponents, axis=1)
        return exponents / row_sums[:, None]
