#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'ldp'
__author__ = 'apple'
__mtime__ = '2019/1/18'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from __future__ import division
from scipy.ndimage.filters import convolve
import numpy as np, itertools
import cv2


def max_values(a):
    b = []
    b.extend(a)
    a.sort(reverse=True)
    c = [0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    for i in range(8):
        if a[0] == b[i]:
            c[i] = 1
            count += 1

        if count == 3:
            return (c[0] * 128) + (c[1] * 64) + (c[2] * 32) + (c[3] * 16) + (c[4] * 8) + (c[5] * 4) + (c[6] * 2) + (
                    c[7] * 1)

    for i in range(8):
        if a[0] == a[1]:
            break
        if a[1] == b[i]:
            c[i] = 1
            count += 1

        if count == 3:
            return (c[0] * 128) + (c[1] * 64) + (c[2] * 32) + (c[3] * 16) + (c[4] * 8) + (c[5] * 4) + (c[6] * 2) + (
                    c[7] * 1)

    for i in range(8):
        if a[2] == a[1]:
            break
        if a[2] == b[i]:
            c[i] = 1
            count += 1

        if count == 3:
            return (c[0] * 128) + (c[1] * 64) + (c[2] * 32) + (c[3] * 16) + (c[4] * 8) + (c[5] * 4) + (c[6] * 2) + (
                    c[7] * 1)
    return (c[0] * 128) + (c[1] * 64) + (c[2] * 32) + (c[3] * 16) + (c[4] * 8) + (c[5] * 4) + (c[6] * 2) + (c[7] * 1)


def ldp(img):
    m0 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    m1 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    m2 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    m3 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    m4 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    m5 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    m6 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    m7 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])

    img0 = cv2.filter2D(img, -1, m0)
    img1 = cv2.filter2D(img, -1, m1)
    img2 = cv2.filter2D(img, -1, m2)
    img3 = cv2.filter2D(img, -1, m3)
    img4 = cv2.filter2D(img, -1, m4)
    img5 = cv2.filter2D(img, -1, m5)
    img6 = cv2.filter2D(img, -1, m6)
    img7 = cv2.filter2D(img, -1, m7)

    ldp = np.zeros(img.shape, np.uint8)
    height, width = img.shape

    for h in range(height):
        for w in range(width):
            ldp[h, w] = max_values(
                [img0[h, w], img1[h, w], img2[h, w], img3[h, w], img4[h, w], img5[h, w], img6[h, w], img7[h, w]])
    return ldp