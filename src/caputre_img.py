#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'caputre_img'
__author__ = 'apple'
__mtime__ = '2018/10/12'
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


def face_detect():
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    num = 201
    while ret:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # 检测人脸
        for (x, y, w, h) in faces:
            roiImg = frame[y:y + w, x:x + h]
            cv2.imwrite("../orig/{0}.bmp".format(num), roiImg)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if num == 220:
                camera.release()
                cv2.destroyAllWindows()
                break
            num += 1
        cv2.imshow('frame', frame)
        ret, frame = camera.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect()