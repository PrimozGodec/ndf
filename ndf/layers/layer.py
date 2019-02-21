from abc import abstractmethod

"""
Layer class that is inherited by all layers
"""


class Layer:

    # each layer has defined number of inputs, `None` if not limited
    number_of_inputs = None

    def __init__(self, name=None):
        self.previous_layers = []
        self.name = name

    def __call__(self, layers):
        if isinstance(layers, list):
            self.previous_layers = layers
        else:
            self.previous_layers.append(layers)
        return self

    def predict(self, inputs, layers_predictions):
        if id(self) in layers_predictions:
            return layers_predictions[id(self)], layers_predictions

        # evaluate up the tree
        prew_predictions = []
        for layer in self.previous_layers:
            pred, layers_predictions = layer.predict(inputs, layers_predictions)
            prew_predictions.append(pred)

        res = self._call_forward(prew_predictions)
        if self.name is not None:
            print(self.name, res.shape)
        layers_predictions[id(self)] = res

        return res, layers_predictions

    def _call_forward(self, inputs):
        if self.number_of_inputs is not None:
            assert len(inputs) == self.number_of_inputs
            return self.forward(inputs[0])
        else:
            assert len(inputs) > 0
            return self.forward(inputs)

    @abstractmethod
    def forward(self, x):
        pass
