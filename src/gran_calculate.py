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


def get_feature_by_img(img, isGray):
    """使用sobel算子计算图片表面梯度,并返回g通道梯度直方图特征和各通道均值及方差
    :param img 读入的图片
    :param isGray 是否是灰度图
    """
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
    if isGray:
        list_features = np.concatenate((hist_g, mean, var))
    else:
        # 计算灰度图均值、方差、直方图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        gray_mean, gray_std = cv2.meanStdDev(gray)
        gray_var = gray_std * gray_std
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        list_features = np.concatenate((hist_g, mean, var, gray_hist, gray_mean, gray_var))

    min_value = np.min(list_features)  # 最大值
    max_value = np.max(list_features)  # 最小值

    # 构建特征字典索引
    features = {}
    for k, v in enumerate(list_features):
        features.setdefault(k + 1, float(v[0] - min_value)/(max_value - min_value))  # 归一化的value
    return features


def process_train_data(isPos, isGray, fname):
    """
    准备训练数据，并将训练数据保存到本地
    :param isPos: 是否是正样本
    :param isGray: 是否是灰度图
    :param fname: 特征值保存的文件名
    :return:
    """
    import os
    from os.path import join
    img_dir = '../orig' if isPos else '../recap'
    features_pieces = []
    for img_name in os.listdir(img_dir):
        # 读入图片文件
        img = cv2.imdecode(np.fromfile(join(img_dir, img_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        features = {0: 1 if isPos else -1}  # 将正例或反例的label加入到字典，正例为1，反例为-1，以便使用svm分类
        features.update(get_feature_by_img(img, isGray))
        features_pieces.append(features)
    # 将特征值存储到本地
    title = features_pieces[0].keys()
    body = [[str(d[t]) if t == 0 else str(t) + ':' + str(d[t]) for t in title] for d in features_pieces]
    s = '\n'.join(['\t'.join([word for word in line]) for line in body])
    with open(fname, 'a') as outfile:
        outfile.write(s)


if __name__ == "__main__":
    process_train_data(True, False, 'positive.txt')  # 准备正样本训练数据
    process_train_data(False, False, 'negative.txt')  # 准备负样本训练数据
