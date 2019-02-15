from ndf.layers import Layer


class Flatten(Layer):

    number_of_inputs = 1

    def forward(self, x):
        nb, h, w, d = x.shape
        return x.reshape(nb, h * w * d)