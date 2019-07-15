import numpy as np
from ndf.layers.layer import Layer


class Flatten(Layer):
    
    number_of_inputs = 1

    def forward(self, x):
        # nb, h, w, d = x.shape
        return x.reshape(x.shape[0], np.prod(x.shape[1:]))


class BatchNorm(Layer):

    number_of_inputs = 1

    def __init__(self, gamma, beta, moving_mu, moving_var, epsilon=0.001, **kwargs):
        self.gamma = gamma
        self.beta = beta

        self.moving_mu = moving_mu
        self.moving_var = moving_var

        self.epsilon = epsilon
        super(BatchNorm, self).__init__(**kwargs)

    def forward(self, x):
        x_norm = (x - self.moving_mu) / np.sqrt(self.moving_var + self.epsilon)
        out = self.gamma * x_norm + self.beta
        return out
