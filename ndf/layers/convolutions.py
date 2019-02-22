import numpy as np
from scipy.signal import convolve2d, fftconvolve
from ndf.layers import Layer


class Conv2D(Layer):

    number_of_inputs = 1

    def __init__(
            self, filters, kernel_size, kernel_weights, bias_weights, strides=1,
            padding="valid", **kwargs):
        self.no_filters = filters
        self.kernel_size = kernel_size
        assert isinstance(strides, int) or strides[0] == strides[1], "Currently only strides of equal numbers supported"
        if isinstance(strides, list) or isinstance(strides, tuple):
            assert len(strides) == 2
            self.stride = strides
        else:
            assert isinstance(strides, int)
            self.stride = (strides, strides)
        self.padding = padding
        self.w = kernel_weights  # [filter_height, filter_width, in_channels, out_channels]
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

        # padding is same: https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
        out_h, out_w = output_shape[1:3]
        filter_h, filter_w = self.kernel_size
        in_h, in_w = input_shape[1:3]
        pad_along_height = max((out_h - 1) * self.stride[0] + filter_h - in_h, 0)
        pad_along_width = max((out_w - 1) * self.stride[1] + filter_w - in_w, 0)
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
            out_h = (in_h - filter_h) // self.stride[0] + 1
            out_w = (in_w - filter_w) // self.stride[1] + 1
        else:  # source: https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
            out_h = int(np.ceil(float(in_h) / float(self.stride[0])))
            out_w = int(np.ceil(float(in_w) / float(self.stride[1])))

        out_shape = (nb_batch, out_h, out_w, self.no_filters)

        # init
        outputs = np.zeros(out_shape)

        # paddings
        pad_top, pad_bottom, pad_left, pad_right = self.padding_size(input_im.shape, out_shape)
        input_im_padded = np.pad(
            input_im, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="constant", constant_values=0)
        _, h, w, _ = input_im_padded.shape

        for b in np.arange(nb_batch):
            for f in np.arange(self.no_filters):

                # source: https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
                view = self.as_stride(input_im_padded[b], self.w.shape[:3], self.stride[0])
                # return numpy.tensordot(aa,kernel,axes=((2,3),(0,1)))
                outputs[b, :, :, f] = np.sum(view * self.w[:, :, :, f], axis=(2, 3, 4))

        return outputs

    @staticmethod
    def as_stride(arr, sub_shape, stride):
        '''Get a strided sub-matrices view of an ndarray.

        <arr>: ndarray of rank 2.
        <sub_shape>: tuple of length 2, window size: (ny, nx).
        <stride>: int, stride of windows.

        Return <subs>: strided window view.

        See also skimage.util.shape.view_as_windows()
        '''
        s0, s1 = arr.strides[:2]
        m1, n1 = arr.shape[:2]
        m2, n2 = sub_shape[:2]

        view_shape = (1 + (m1 - m2) // stride, 1 + (n1 - n2) // stride, m2,
                      n2) + arr.shape[2:]
        strides = (stride * s0, stride * s1, s0, s1) + arr.strides[2:]
        subs = np.lib.stride_tricks.as_strided(arr, view_shape,
                                               strides=strides)

        return subs
