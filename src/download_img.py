#!/usr/bin/env python
# coding=utf-8

"""
__title__ = 'download'
__author__ = 'apple'
__mtime__ = '2018/8/2'
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
import re
import requests
from bs4 import BeautifulSoup
from os.path import join
from PIL import Image
from os import remove


def dowmloadPic(html, savepath):
    # print(html)
    soup = BeautifulSoup(html, 'html.parser')
    trs = soup.find_all('tr')
    # print(trs)
    i = 1
    for tr in trs:
        img_info = tr.find_all('td')[-1].text
        img_name = re.findall(r"ImageName = (.+?)  URL", img_info)[0]  #
        img_url = re.findall(r"URL = (.+?)  Category", img_info)[0]  #
        print('正在下载第' + str(i) + '张图片，图片地址:' + img_url)
        try:
            pic = requests.get(img_url, timeout=10000)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue
        except requests.exceptions.ReadTimeout:
            print('【错误】请求超时')
            continue

        image_path = join(savepath, img_name)
        fp = open(image_path, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1
        filterimg(image_path)


def filterimg(img_name):
    fp = open(img_name, 'rb')
    count = fp.readline(1)
    try:
        img = Image.open(fp)
    except OSError:
        return
    x, y = img.size
    fp.close()
    if b'\xff' not in count: #  删除空白图片
        remove(img_name)
    else:
        pass
        # if x > x


if __name__ == '__main__':
    url = 'http://www.ee.columbia.edu/~dvmmweb/dvmm/downloads/PIM_PRCG_dataset/techreport_recapturedCG.html'
    result = requests.get(url)
    dowmloadPic(result.text, savepath='../recapture')