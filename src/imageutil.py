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
import numpy as np
import os


def face_detect():
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    num = 301
    while ret:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # 检测人脸
        for (x, y, w, h) in faces:
            roiImg = frame[y:y + w, x:x + h]
            resize_img = cv2.resize(roiImg, (100, 100))
            cv2.imwrite("../orig/{0}.bmp".format(num), resize_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if num == 350:
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


def resize(path, dst=None):
    import os
    import numpy as np
    from os.path import join
    for img_name in os.listdir(path):
        # 读入图片文件
        img = cv2.imdecode(np.fromfile(join(path, img_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        try:
            resize_img = cv2.resize(img, (100, 100))
            if dst is None:
                cv2.imwrite(join(path, img_name), resize_img)
            else:
                cv2.imwrite(join(dst, img_name), resize_img)
        except Exception as e:
            print(e)
            continue
        # cv2.imshow('img', img)
        # cv2.imshow('resize', resize_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    if image is None:
        return
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, contours, _ = cv2.findContours(gray, 2, 2)
    # angle = 0
    # for cnt in contours:
    #     # 最小的外接矩形
    #     theta = cv2.minAreaRect(cnt)[2]
    #     if abs(theta) > abs(angle):
    #         angle = theta
    # print("angle:%d" % angle)
    (h, w) = image.shape[:2]
    print('%dx%d开始旋转' % (h, w))
    if h >= w:
        return image
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def rotate(angle):
    dir = input('dir:')
    outdir = input('out:')
    for img in os.listdir(dir):
        print(img)
        image = cv2.imdecode(np.fromfile(os.path.join(dir, img), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        cv2.imwrite(os.path.join(outdir, img), rotate_bound(image, angle))


def cap_face():
    dir = input('images dir:')
    outdir = input('out:')
    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    for img in os.listdir(dir):
        print(img)
        image = cv2.imdecode(np.fromfile(os.path.join(dir, img), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 10)  # 检测人脸
        num = 1
        for (x, y, w, h) in faces:
            roiImg = image[y:y + w, x:x + h]
            if roiImg.shape[0] < 100 or roiImg.shape[1] < 100:
                resize_img = roiImg
            else:
                resize_img = cv2.resize(roiImg, (100, 100))
            cv2.imwrite(os.path.join(outdir, img), resize_img)


if __name__ == "__main__":
    # face_detect()
    # rotate(90)
    # resize('../rotate', '../scale_img')
    cap_face()