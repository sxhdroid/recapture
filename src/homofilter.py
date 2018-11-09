#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'homofilter'
__author__ = 'apple'
__mtime__ = '2018/11/8'
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
import time


def homo(img, high, low, c):
    """
    :param img:  bgr原图
    :param high: 高频增益
    :param low:  低频增益
    :param c:  锐化程度
    :return:  增强后的bgr图片
    """
    start = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    Img = v.astype(np.double)
    H = np.zeros(v.shape, dtype=np.double)
    (Height, Width) = v.shape  # 返回的行数和列数
    sigma = max(Height, Width)

    CenterX = np.floor(Width / 2)  # 中心点坐标
    CenterY = np.floor(Height / 2)

    LogImg = np.log(Img + 1)  # 图像对数数据
    Log_FFT = np.fft.fft2(LogImg)  # 傅里叶变换
    Log_FFT = np.fft.fftshift(Log_FFT)  # FFT的结果进行中心化

    for Y in range(Height):
        for X in range(Width):
            Dist = (X - CenterX) * (X - CenterX) + (Y - CenterY) * (Y - CenterY)  # 点（X, Y）到频率平面原点的距离
            H[Y, X] = (high - low) * (1 - np.exp(-c * (Dist / (2 * sigma * sigma)))) + low  # 同态滤波器函数

    Log_FFT = H * Log_FFT  # 滤波，矩阵点乘
    Log_FFT = np.fft.ifftshift(Log_FFT)  # FFT的结果进行中心化
    Log_FFT = np.fft.ifft2(Log_FFT)  # 反傅立叶变换

    Out = np.exp(Log_FFT) - 1  # 取指数
    # 指数处理ge = exp(g) - 1; % 归一化到[0, L - 1]
    Max = np.max(Out)
    Min = np.min(Out)
    Range = Max - Min
    ImageOut = np.zeros(v.shape, dtype=np.uint8)
    for Y in range(Height):
        for X in range(Width):
            ImageOut[Y, X] = np.uint8(255 * (Out[Y, X] - Min) / Range)
    HSV = cv2.merge([h, s, ImageOut])
    outimg = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    end = time.time()
    print('time is ' + str(end - start))
    return outimg


if __name__ == '__main__':
    src = cv2.imread('../images/0004_01_01_03_372.jpg', cv2.IMREAD_UNCHANGED)
    dst = homo(src, 2, 0.2, 0.1)
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



