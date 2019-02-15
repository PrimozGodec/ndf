import numpy as np
from ndf.layers import Layer


def get_im2col_indices(x_shape, field_height, field_width, padding=1,
                       stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height,
                                              field_width,
                                              padding,
                                              stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C,
                                           -1)
    return cols


class MaxPooling2D(Layer):

    number_of_inputs = 1

    def __init__(self, pool_size, strides=1):
        self.pool_size = pool_size
        self.stride = strides
        super(MaxPooling2D, self).__init__()

    def forward(self, x):
        x = x[0]  # this layer has only one input
        # source: https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
        nb, h, w, d = x.shape
        h_out, w_out = x.shape[1:3] / self.stride

        x_tranposed = x.transpose(0, 3, 1, 2)
        x_reshaped = x_tranposed.reshape(nb * d, 1, h, w)
        X_col = im2col_indices(x_reshaped, self.pool_size[0], self.pool_size[1], padding=0, stride=self.stride)

        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        out = out.reshape(h_out, w_out, nb, d)

        return out.transpose(2, 0, 1, 3)

class AveragePooling2D(Layer):

    number_of_inputs = 1

    def __init__(self, pool_size, strides=1):
        self.pool_size = pool_size
        self.stride = strides
        super(AveragePooling2D, self).__init__()

    def forward(self, x):
        x = x[0]  # this layer has only one input
        # source: https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
        nb, h, w, d = x.shape
        h_out, w_out = x.shape[1:3] / self.stride

        x_tranposed = x.transpose(0, 3, 1, 2)
        x_reshaped = x_tranposed.reshape(nb * d, 1, h, w)
        X_col = im2col_indices(x_reshaped, self.pool_size[0], self.pool_size[1], padding=0, stride=self.stride)

        out = np.mean(X_col, axis=0)
        out = out.reshape(h_out, w_out, nb, d)

        return out.transpose(2, 0, 1, 3)

