import numpy as np
from ndf.layers import Layer
from ndf.utils.image_to_col import im2col_indices


class Conv2D(Layer):

    number_of_inputs = 1

    def __init__(
            self, filters, kernel_size, kernel_weights, bias_weights, strides=1,
            padding="valid", **kwargs):
        self.no_filters = filters
        self.kernel_size = kernel_size
        assert isinstance(strides, int) or strides[0] == strides[1], \
            "Currently only strides of equal numbers supported"
        if isinstance(strides, list) or isinstance(strides, tuple):
            assert len(strides) == 2
            self.stride = strides
        else:
            assert isinstance(strides, int)
            self.stride = (strides, strides)
        self.padding = padding
        self.w = kernel_weights  # [filter_h, filter_w, in_chan, out_chans]
        self.b = bias_weights
        self.previous_layer = None
        self.next_layers = []
        super(Conv2D, self).__init__(**kwargs)

    def padding_size(self, input_shape, output_shape):
        """
        Function computes the sizes to pad with0
        """
        if self.padding == "valid":
            return 0, 0, 0, 0

        # padding is same:
        # https://stackoverflow.com/questions/45254554/
        # tensorflow-same-padding-calculation
        out_h, out_w = output_shape[1:3]
        filter_h, filter_w = self.kernel_size
        in_h, in_w = input_shape[1:3]
        pad_along_height = max(
            (out_h - 1) * self.stride[0] + filter_h - in_h, 0)
        pad_along_width = max((out_w - 1) * self.stride[1] + filter_w - in_w, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_top, pad_bottom, pad_left, pad_right

    def forward(self, input_im):
        nb_batch, in_h, in_w, in_depth = input_im.shape
        filter_h, filter_w = self.kernel_size

        # source: https://stackoverflow.com/questions/45254554/
        # tensorflow-same-padding-calculation
        if self.padding == "valid":
            out_h = (in_h - filter_h) // self.stride[0] + 1
            out_w = (in_w - filter_w) // self.stride[1] + 1
        else:
            out_h = int(np.ceil(float(in_h) / float(self.stride[0])))
            out_w = int(np.ceil(float(in_w) / float(self.stride[1])))

        out_shape = (nb_batch, out_h, out_w, self.no_filters)

        # paddings
        paddings = self.padding_size(input_im.shape, out_shape)

        # source: https://wiseodd.github.io/techblog/2016/07/16/
        # convnet-conv-layer/

        x = input_im.transpose(0, 3, 1, 2)  # to channel first
        x_col = im2col_indices(
            x, self.kernel_size[0], self.kernel_size[1],
            paddings=paddings, stride=self.stride[0])

        w_col = self.w.transpose(3, 2, 0, 1).reshape(self.no_filters, -1)

        out = w_col @ x_col + self.b[:, None]

        out = out.reshape(self.no_filters, out_h, out_w, nb_batch)
        out = out.transpose(3, 1, 2, 0)

        return out
