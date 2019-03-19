import os
import time

import numpy as np
from PIL import Image

from ndf.example_models import squeezenet

"""
This is an example script that show how squeeze-net can be used
"""

# example of use
if __name__ == "__main__":
    def preprocess_squeezenet(image):
        mean_pixel = [104.006, 116.669, 122.679]
        image = np.array(image, dtype=float)
        if len(image.shape) < 4:
            image = image[None, ...]
        swap_img = np.array(image)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out - mean_pixel


    bs = 1
    sn = squeezenet(include_softmax=False)
    _sum = 0
    for file in os.listdir("/Users/primoz/Desktop/test_images/yplp/nucleus/")[:20]:
        if file.endswith(".jpg"):
            path = os.path.join("/Users/primoz/Desktop/test_images/yplp/nucleus/", file)

            i = Image.open(path)
            i = i.resize((227, 227))
            im = np.array(i)[None, ...]

            im1 = preprocess_squeezenet(im)

            im_r = np.repeat(im1, bs, axis=0)

            t = time.time()
            p = sn.predict([im_r])
            ttt = (time.time() - t)

            _sum += ttt
    print(_sum / (bs * 20))
