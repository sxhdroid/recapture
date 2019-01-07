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
import time
from matplotlib import pyplot as plt
from skimage.feature import hog


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


def resize(path, dst=None, size=(128, 128)):
    import os
    import numpy as np
    from os.path import join
    for img_name in os.listdir(path):
        # 读入图片文件
        img = cv2.imdecode(np.fromfile(join(path, img_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        try:
            resize_img = cv2.resize(img, (size[0], size[1]))
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


def detect_face_and_resize(path, dst=None, size=(128, 128)):
    import os
    import numpy as np
    from os.path import join

    # 加载人脸检测过滤器
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    for img_name in os.listdir(path):
        # 读入图片文件
        img = cv2.imdecode(np.fromfile(join(path, img_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)  # 检测人脸
        print('%s faces is %d' % (img_name, len(faces)))
        for (x, y, w, h) in faces:
            roiImg = img[y:y + w, x:x + h]
            try:
                resize_img = cv2.resize(roiImg, (size[0], size[1]))
                if dst is None:
                    cv2.imwrite(join(path, img_name), resize_img)
                else:
                    cv2.imwrite(join(dst, img_name), resize_img)
            except Exception as e:
                print(e)
                continue


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


# # sobel算子的实现
def sobel(img):
    # 计算原图的表面梯度
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def direction_of_8_sobel(img):
    # start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (height, width) = img.shape
    # new_image = np.zeros(img.shape, dtype=np.uint8)
    kernel_0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    kernel_90 = np.array([[1, 0, -1], [-2, 0, 2], [1, 0, -1]])
    kernel_135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    kernel_180 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_225 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    kernel_270 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_315 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

    # max_kernel[7] = abs(np.sum(img[i:i + 3, j:j + 3] * kernel_315))

    dst_0 = cv2.filter2D(img, -1, kernel_0)
    dst_45 = cv2.filter2D(img, -1, kernel_45)
    dst_90 = cv2.filter2D(img, -1, kernel_90)
    dst_135 = cv2.filter2D(img, -1, kernel_135)
    dst_180 = cv2.filter2D(img, -1, kernel_180)
    dst_225 = cv2.filter2D(img, -1, kernel_225)
    dst_270 = cv2.filter2D(img, -1, kernel_270)
    dst_315 = cv2.filter2D(img, -1, kernel_315)

    all_values = np.array([dst_0.flatten(), dst_45.flatten(), dst_90.flatten(), dst_135.flatten()
                              , dst_180.flatten(), dst_225.flatten(), dst_270.flatten(), dst_315.flatten()])
    max_values_index = all_values.argmax(axis=0)  # 每列最大数索引
    max_value = [0] * len(max_values_index)
    j = 0
    for i in max_values_index:  # 每列最大值索引
        max_value[j] = all_values[i][j]
        j += 1
    new_image = np.reshape(max_value, newshape=(height, width))  # 转换为图片数值矩阵
    # step1 = time.time()
    # print("step1:%f" % (start - step1))
    # cv2.imshow('8', np.hstack((img, dst_0, dst_45, dst_90, dst_135, dst_180, dst_225, dst_270, dst_315, new_image)))
    # cv2.waitKey(0)
    return new_image


if __name__ == "__main__":
    # face_detect()
    # rotate(90)
    # resize('../orig', '../orig')
    # detect_face_and_resize('../orig', '../orig')
    # cap_face()
    # 计算原图的表面梯度
    img = cv2.imread('../orig/0001_00_00_01_12.jpg', cv2.IMREAD_UNCHANGED)
    direction_of_8_sobel(img)
    # cv2.imshow('normal', np.sqrt(img/float(np.max(img))))
    # sobel(img)
    # sobel_8_or(img)
    # cv2.imshow('sss', direction_of_8_sobel(img))
    # dst1 = sobel(img)
    # hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
    # plt.hist(hist1, facecolor='black')
    # cv2.imshow('1', dst1)
    #
    # from src import homofilter
    # homo_dst = homofilter.homo(img, 2, 0.2, 0.1)
    # sobel_8_or(homo_dst, '111')
    # start = time.time()
    # dst2 = kernelction_of_8_sobel(homo_dst)
    # print(time.time()-start)
    # plt.hist(cv2.calcHist([dst2], [0], None, [256], [0, 256]))
    # cv2.imshow('2', dst2)
    # plt.show()

    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(x)  # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    #
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)

    # fd = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(4, 4))
    # print(len(fd))