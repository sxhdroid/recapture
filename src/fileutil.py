#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'rename_file'
__author__ = 'apple'
__mtime__ = '2018/10/15'
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
import os
import shutil
import sys


def rename(fd, pre):
    count = 0
    for dirname, dirnames, filenames in os.walk(fd):
        print('dir: %s' % dirname)
        for filename in filenames:
            print('filename: %s' % filename)
            try:
                os.rename(os.path.join(dirname, filename), os.path.join(dirname, pre + '_' + str(count) + ".jpg"))
            except FileExistsError as e:
                continue
            finally:
                count += 1


def copy(fd, dst):
    for dirname, dirnames, filenames in os.walk(fd):
        for filename in filenames:
            try:
                shutil.copy(os.path.join(dirname, filename), os.path.join(dst, filename))
            except FileExistsError as e:
                print(e)
                continue


if __name__ == "__main__":
    # rename(input('path:'), pre='jk')
    copy(input('src:'), input('dst:'))