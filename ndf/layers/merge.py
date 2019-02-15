import numpy as np
from ndf.layers import Layer

class Concatenate(Layer):

    number_of_inputs = None

    def __init__(self, axis=-1):
        self.axis = axis
        super(Concatenate, self).__init__()

    def forward(self, x):
        return np.concatenate(x, axis=self.axis)