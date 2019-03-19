import unittest
import os
import inspect

from PIL import Image
import numpy as np

from ndf.example_models import squeezenet


class TestNdf(unittest.TestCase):

    def setUp(self):
        self.sn = squeezenet(include_softmax=False)

    def test_squeezenet(self):
        """
        While testing squeezenet we test whole framework
        """
        bs = 4  # batch size

        path = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))

        i = Image.open(os.path.join(path, "book-2020460_640.jpg"))
        i = i.resize((227, 227))
        im = np.array(i)[None, ...]

        im_r = np.repeat(im, bs, axis=0)

        p = self.sn.predict([im_r])
        self.assertListEqual(list(p[0].shape), [4, 1000])
        self.assertEqual(len(p), 1)


if __name__ == '__main__':
    unittest.main()
