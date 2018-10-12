#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'mysvm'
__author__ = 'apple'
__mtime__ = '2018/10/11'
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
from svmutil import *
from src.gran_calculate import get_feature_by_img
import cv2


def train():
    y_pos, x_pos = svm_read_problem('positive.txt')
    y_neg, x_neg = svm_read_problem('negative.txt')
    y_pos_t_len = int(len(y_pos) * 0.8)
    y_neg_t_len = int(len(y_neg) * 0.8)
    x_pos_t_len = int(len(x_pos) * 0.8)
    x_neg_t_len = int(len(x_neg) * 0.8)

    y_train = y_pos[: y_pos_t_len] + y_neg[: y_neg_t_len]  # 训练集
    x_train = x_pos[: x_pos_t_len] + x_neg[: x_neg_t_len]  # 训练集

    y_predict = y_pos[y_pos_t_len:] + y_neg[y_neg_t_len:]  # 预测集
    x_predict = x_pos[x_pos_t_len:] + x_neg[x_neg_t_len:]  # 预测集

    model = svm_train(y_train[:], x_train[:], '-c 5')
    svm_save_model('recap.md', model)
    p_label, p_acc, p_val = svm_predict(y_predict[:], x_predict[:], model)
    # print(p_label)
    # print(p_acc)
    # print(p_val)


def predict(y, x, model):
    p_label, p_acc, p_val = svm_predict(y, x, model)
    # if p_label[0] == 1:
    #     print('真人')
    # else:
    #     print('非真人')
    return p_label[0]


def face_detect():
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    model = svm_load_model('recap.md')  # 读取预测模型
    num = 201
    while ret:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # 检测人脸
        for (x, y, w, h) in faces:
            roiImg = frame[y:y + w, x:x + h]
            # gray = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
            feature = get_feature_by_img(roiImg, False)
            label = predict([-1], [feature], model)
            # if label == -1 and num <= 220:  # 用来抓取误识图片
            #     cv2.imwrite("../orig/{0}.bmp".format(num), roiImg)
            #     num += 1
            cv2.putText(frame, '1' if label == 1 else '-1', (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5, color=(0, 255, 0), thickness=2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        ret, frame = camera.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # train()
    face_detect()