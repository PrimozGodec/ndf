import numpy as np
from ndf.layers import Layer


class Conv2D(Layer):

    number_of_inputs = 1

    def __init__(
            self, filters, kernel_size, kernel_weights, bias_weights, stride=1, padding="valid"):
        self.no_filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.w = kernel_weights  # [filter_height, filter_width, in_channels, out_channels]
        self.b = bias_weights
        self.previous_layer = None
        self.next_layers = []
        super(Conv2D, self).__init__()


    def padding_size(self, input_shape, output_shape):
        """
        Function computes the sizes to pad with0
        """
        if self.padding == "valid":
            return 0, 0, 0, 0

        # padding is same: https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
        out_h, out_w = output_shape[2:]
        filter_h, filter_w = self.kernel_size
        in_h, in_w = input_shape[2:]
        pad_along_height = max((out_h - 1) * self.stride + filter_h - in_h, 0)
        pad_along_width = max((out_w - 1) * self.stride + filter_w - in_w, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_top, pad_bottom, pad_left, pad_right

    def forward(self, input_im):
        nb_batch, in_h, in_w, in_depth = input_im.shape
        filter_h, filter_w = self.kernel_size

        #output shape
        if self.padding == "valid":
            height = (in_h - filter_h) // self.stride + 1
            width = (in_w - filter_w) // self.stride + 1
        else:  # source: https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
            height = np.ceil(float(in_h) / float(self.stride))
            width = np.ceil(float(in_w) / float(self.stride))

        out_shape = (nb_batch, self.no_filters, height, width)
        out_h, out_w = out_shape[1:3]

        # init
        outputs = np.zeros((nb_batch, out_h, out_w))

        # paddings
        pad_top, pad_bottom, pad_left, pad_right = self.padding_size(input_im.shape, out_shape)

        # convolution operation
        for x in np.arange(nb_batch):
            for y in np.arange(self.no_filters):
                for h in np.arange(out_h + pad_top + pad_bottom):
                    for w in np.arange(out_w + pad_left + pad_right):
                        h_shift, w_shift = h * self.stride - pad_top, w * self.stride - pad_left

                        patch = input_im[x,
                                max(h_shift, 0): min(h_shift + filter_h, in_h),
                                max(w_shift, 0): min(w_shift + filter_w, in_w)]
                        outputs[x, h, w, y] = np.sum(
                            patch * self.w[
                                    max(0, pad_top - h): filter_h - max(0, h - in_h),
                                    max(0, pad_left - w): filter_w - max(0, w - in_w),
                                    y]) + self.b[y]
        return outputs
