import unittest
import os
import inspect

import numpy as np

from ndf.example_models import encoder


class TestEnc(unittest.TestCase):

    def setUp(self):
        self.en = encoder()

    def test_encoder(self):
        """
        While testing encoder we test added conv1d, dense and batch norm layers
        """

        bn = 4

        i = np.array([[[float(i == (j % 35)) for i in range(35)] for j in range(120)]])
        i_r = np.repeat(i, bn, axis=0)

        p = self.en.predict([i_r])
        self.assertListEqual(list(p[0].shape), [4, 196])
        self.assertEqual(len(p), 1)


if __name__ == '__main__':
    unittest.main()

