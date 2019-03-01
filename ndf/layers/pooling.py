import numpy as np
from ndf.layers import Layer
from ndf.utils.image_to_col import im2col_indices


class MaxPooling2D(Layer):

    number_of_inputs = 1

    def __init__(self, pool_size, strides=1, **kwargs):
        self.pool_size = pool_size
        if isinstance(strides, list) or isinstance(strides, tuple):
            assert len(strides) == 2
            assert strides[0] == strides[1], \
                "We are currently supporting only strides with width and height"
            self.stride = strides[0]
        else:
            assert isinstance(strides, int)
            self.stride = strides
        super(MaxPooling2D, self).__init__(**kwargs)

    def forward(self, x):
        """
        source: https://wiseodd.github.io/techblog/2016/07/18/
        convnet-maxpool-layer/
        """
        nb, h, w, d = x.shape
        h_out = (h - self.pool_size[0]) // self.stride + 1
        w_out = (w - self.pool_size[1]) // self.stride + 1

        x_tranposed = x.transpose(0, 3, 1, 2)
        x_reshaped = x_tranposed.reshape(nb * d, 1, h, w)

        x_col = im2col_indices(x_reshaped, self.pool_size[0], self.pool_size[1],
                               paddings=(0, 0, 0, 0), stride=self.stride)

        out = np.max(x_col, axis=0)
        out = out.reshape((h_out, w_out, nb, d))

        return out.transpose(2, 0, 1, 3)


class AveragePooling2D(Layer):

    number_of_inputs = 1

    def __init__(self, pool_size, strides=1, **kwargs):
        self.pool_size = pool_size
        self.stride = strides
        super(AveragePooling2D, self).__init__(**kwargs)

    def forward(self, x):
        """
        source: https://wiseodd.github.io/techblog/2016/07/18/
        convnet-maxpool-layer/
        """
        nb, h, w, d = x.shape
        h_out = (h - self.pool_size[0]) // self.stride + 1
        w_out = (w - self.pool_size[1]) // self.stride + 1

        x_tranposed = x.transpose(0, 3, 1, 2)
        x_reshaped = x_tranposed.reshape(nb * d, 1, h, w)
        x_col = im2col_indices(x_reshaped, self.pool_size[0], self.pool_size[1],
                               paddings=(0, 0, 0, 0), stride=self.stride)

        out = np.mean(x_col, axis=0)
        out = out.reshape((h_out, w_out, nb, d))

        return out.transpose(2, 0, 1, 3)
