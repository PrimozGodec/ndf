from abc import abstractmethod

"""
Layer class that is inherited by all layers
"""


class Layer:

    # each layer has defined number of inputs, `None` if not limited
    number_of_inputs = None

    def __init__(self):
        self.previous_layers = []

    def __cal__(self, layers):
        self.previous_layers = layers

    def predict(self, inputs, layers_predictions):
        if id(self) in layers_predictions:
            return layers_predictions[id(self)], layers_predictions

        # evaluate up the tree
        prew_predictions = []
        for layer in self.previous_layers:
            pred,layers_predictions = layer.predict()
            prew_predictions.append(pred)

        res = self.forward(prew_predictions)
        prew_predictions[id(self)] = res

        return res, prew_predictions

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
