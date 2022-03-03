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
    #image = cv2.imread(args.image, 0)
    orig = cv2.imread(args.image)
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    f = np.fft.fft2(image)
    #cv2.imwrite('fft_abs.png', np.abs(f))

    f_shift = np.fft.fftshift(f)
    #cv2.imwrite('fft_shift.png', np.abs(f_shift))

    spectrum_1 = 20 * np.log(np.abs(f_shift))
    spectrum_2 = 20 * np.log(np.abs(f))

    L = max(image.shape)
    freq = np.fft.fftfreq(L)[:int(L/2)]
    dist = np.sqrt(np.fft.fftfreq(image.shape[0])[:, np.newaxis]**2 + np.fft.fftfreq(image.shape[1])**2)
    nums = np.histogram(dist.ravel(), bins=freq)[0]
    hist, bins = np.histogram(dist.ravel(), bins=freq, weights=spectrum_2.ravel())
    centers = (bins[:-1] + bins[1:]) / 2

    plt.subplot(131), plt.imshow(color)
    plt.title('image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(spectrum_1, cmap = 'gray')
    plt.title('magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.plot(centers, hist / nums)
    plt.title('histogram'), plt.xlabel('frequency'), plt.ylabel('spectrum')
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


