#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'gran_calculate'
__author__ = 'apple'
__mtime__ = '2018/9/12'
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
import cv2
import numpy as np
from matplotlib import pyplot as plt
import svmutil


def get_feature(img_path):
    """使用sobel算子计算图片表面梯度,并返回g通道梯度直方图特征和各通道均值及方差"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # 计算原图的表面梯度
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 计算原图的均值、方差
    mean, std = cv2.meanStdDev(dst)  # 计算均值和标准差
    var = std * std  # 方差
    # print(var)
    # 计算原图g通道梯度直方图特征
    hist_g = cv2.calcHist([dst], [1], None, [32], [0, 256])  # 获取g通道梯度直方图特征

    # 计算灰度图均值、方差、直方图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    gray_mean, gray_std = cv2.meanStdDev(gray)
    gray_var = gray_std * gray_std
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])

    # 返回格式：原图g通道直方图，原图梯度均值， 原图方差， 灰色图直方图，灰度图均值，灰度图方差
    return hist_g, mean, var, gray_hist, gray_mean, gray_var


if __name__ == "__main__":
    # 读入图片文件
    g_hist, gand_mean, gand_var = get_feature("../images/0804-一寸证件照_50119.jpg")
    print(g_hist)
    # plt.plot(g_hist, color='blue')
    # plt.xlim([0, 256])
    # plt.show()
    # sobel("../orig/1.bmp")
