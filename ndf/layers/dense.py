import numpy as np
from ndf.layers.layer import Layer


class Dense(Layer):
    
    number_of_inputs = 1

    def __init__(self, kernel_weights, bias_weights, **kwargs):
        self.w = kernel_weights
        self.b = bias_weights
        super(Dense, self).__init__(**kwargs)

    def forward(self, x):
        return np.dot(x, self.w) + self.b
