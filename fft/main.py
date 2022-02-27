#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: fenc=utf-8 ff=unix ft=python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image')

def main(args):
    image = cv2.imread(args.image, 0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f = np.fft.fft2(image)
    #cv2.imwrite('fft_abs.png', np.abs(f))

    f_shift = np.fft.fftshift(f)
    #cv2.imwrite('fft_shift.png', np.abs(f_shift))

    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    plt.subplot(121), plt.imshow(image, cmap = 'gray')
    plt.title('image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


