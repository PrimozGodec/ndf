from ndf.layers import Layer


class Input(Layer):

    number_of_inputs = 1

    def __init__(self, shape):
        self.shape = shape
        super(Input, self).__init__()

    def predict(self, inputs, layers_predictions):
        return inputs[id(self)], layers_predictions