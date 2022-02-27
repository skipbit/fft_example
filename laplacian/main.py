#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: fenc=utf-8 ff=unix ft=python

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image')

def main(args):
    image = cv2.imread(args.image, 0)

    l = cv2.Laplacian(image, cv2.CV_64F)
    print('laplacian value = {}'.format(l.var()));


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


