import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, paddings,
                       stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    p_t, p_b, p_l, p_r = paddings

    out_height = (H + p_t + p_b - field_height) // stride + 1
    out_width = (W + p_t + p_b - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, paddings=(1, 1, 1, 1), stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p_t, p_b, p_l, p_r = paddings
    x_padded = np.pad(x, ((0, 0), (0, 0), (p_t, p_b), (p_l, p_r)),
                      mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height,
                                              field_width,
                                              paddings,
                                              stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols