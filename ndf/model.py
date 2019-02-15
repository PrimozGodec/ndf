class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, inputs):
        # dictionary to hold predictions of layers in
        # case they are required again
        layer_predictions = {}
        assert len(inputs) == len(self.inputs)
        inputs_pairs = {id(x): y for x, y in zip(self.inputs, inputs)}
        predictions = []

        for output in self.outputs:
            pred, layer_preds = output.predict(inputs_pairs, layer_predictions)
            # merge previous layer activations dict with new one
            layer_predictions = {**layer_predictions, **layer_preds}
            predictions.append(pred)

        return predictions


