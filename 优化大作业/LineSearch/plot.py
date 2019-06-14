#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: plot.py
@time: 2019/06/07
@contact: 1284975112@qq.com
@site:
@software: PyCharm

"""

import numpy as np
import matplotlib.pyplot as plt


def fun(x1, x2):
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)


def grad(x1, x2):
    grad_x1 = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1)
    grad_x2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
    gradd = np.array([grad_x1, grad_x2])
    return gradd


# 计算2范数
def norm2(arr):
    return np.sqrt(np.sum(arr ** 2))


if __name__ == '__main__':

    # 提前计算好结果
    result = 2.55926669666
    alpha = np.linspace(0.1, 0.4, 4)
    beta = np.linspace(0.1, 0.9, 9)
    row = np.size(alpha)
    col = np.size(beta)
    fig, ax = plt.subplots(row, col)
    # 用于计数
    k = 0
    for i in range(row):
        for j in range(col):
            # 1.给定初始点
            x = np.array([0.1, 0.1])

            value = np.array([])
            timer = np.array([])

            # 2.进入判断条件
            stop_value = 1e-7
            grad_x = grad(x[0], x[1])
            norm2_grad = norm2(grad_x)
            while norm2_grad >= stop_value:
                # 3. 给定负梯度方向为下降方向
                dk = -grad_x
                dk_t = np.array([[dk[0]], [dk[1]]])
                # 4. 回溯直线搜索 给α,β
                tk = 1
                xk_1 = x + tk * dk
                while fun(xk_1[0], xk_1[1]) > fun(x[0], x[1]) + alpha[i] * tk * (grad_x @ dk_t):
                    tk = beta[j] * tk
                    xk_1 = x + tk * dk
                # 5.赋值回2
                x = xk_1
                grad_x = grad(x[0], x[1])
                norm2_grad = norm2(grad_x)

                # 添加方便做图
                k = k+1
                if fun(x[0], x[1])-result < 1e-5:
                    timer = np.append(timer, k)
                    value = np.append(value, fun(x[0], x[1])-result)
            ax[i][j].plot(timer, value)
            ax[i][j].set_title('alpha=' + str(alpha[i]) + ',beta=' + str(beta[j]), color='r')

    plt.show()
