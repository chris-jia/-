#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: svm.py
@time: 2019/06/06
@contact: 1284975112@qq.com
@site:
@software: PyCharm

"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
import _pickle as pickle


# 将mnist数据集转为images和labels矩阵
# 参考：https://blog.csdn.net/simple_the_best/article/details/75267863
def load_mnist(path, kind='train'):
    """加载mnist集"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# 打印数字的图片以验证
def print_image(n,images,labels):
    fig, ax = plt.subplots(
        nrows=int(np.sqrt(n))+1,
        ncols=int(np.sqrt(n))+1,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(n):
        img = images[i].reshape(28, 28)
        ax[i].set_title(labels[i], color='r')
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == '__main__':
    images, labels = load_mnist('')
    imagesTest, lablesTest = load_mnist('', kind='t10k')
    # 已经建立模型 此处无需再计算
    # model = svm.LinearSVC()
    # model.fit(images, labels)
    # #保存模型
    # with open('./model.pkl', 'wb') as file:
    #     pickle.dump(model, file)

    # 恢复模型
    with open('./model.pkl', 'rb') as file:
        model = pickle.load(file)

    # 先预测前三十个
    result = model.predict(imagesTest[:30, :])
    # 打印显示
    print_image(30, imagesTest, result[:30])

    #计算总体正确率
    result = model.predict(imagesTest)
    correct_rate = np.sum(result == lablesTest)/result.size
    print('测试集总数：', result.size)
    print('测试准确率：', correct_rate)
