#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: question1.py
@time: 2019/06/08
@contact: 1284975112@qq.com
@site:
@software: PyCharm

"""
import numpy as np

# 有可行初始解

# 产生0,10之间的随机数
def generateA(p, n):
    return np.random.rand(p, n)*10

# 产生0,1的x_hat
def generate_x_hat(n):
    return np.random.rand(n, 1)

def fun(x):
    result = 0
    for i in x:
        result = i*np.log(i) + result
    return result

def calc_grad(x):
    return  np.log(x) + 1

def calc_hessian(x):
    len = x.size
    result = np.zeros((len, len))
    for i in range(len):
        result[i][i] = 1/x[i]
    return result

def calc_dk(hessian, A, grad, p):
    len = grad.size
    zero1 = np.zeros((p, p))
    zero2 = np.zeros(p)
    neg_grad = -grad

    # 进行拼接组合
    A_col1 = np.concatenate((hessian, A), axis=0)
    A_t = A.T
    A_col2 = np.concatenate((A_t, zero1), axis=0)
    A_new = np.concatenate((A_col1, A_col2), axis=1)
    b_new = np.concatenate((neg_grad, zero2), axis=0)

    # 计算
    x_new = np.linalg.solve(A_new, b_new)
    dk = x_new[:len]
    w = x_new[len:]
    return dk, w

def calc_lambda_k2(dk, hessian):
    dk_t = dk.T
    lambda_k2 = np.dot(np.dot(dk, hessian), dk_t)
    return lambda_k2



if __name__ == '__main__':
     p = 30
     n = 100
     # 已保存
     # A = generateA(p, n)
     # x_hat = generate_x_hat(n)
     # np.savetxt('matrix_A.txt', A)
     # np.savetxt('matrix_x_hat.txt', x_hat)

     # 读取 设初值
     A = np.loadtxt('matrix_A.txt')
     x_hat = np.loadtxt('matrix_x_hat.txt')
     b = np.dot(A, x_hat)

     # 1.给定x_hat为初始点

     x = x_hat
     error_value = 1e-25

     # 2.确定牛顿方向和牛顿减少量
     # 进行矩阵拼接

     hessian = calc_hessian(x)
     grad = calc_grad(x)

     dk,w = calc_dk(hessian, A, grad, p)
     lambda_k2 = calc_lambda_k2(dk, hessian)

     # 3.停止准则
     while lambda_k2/2 > error_value :
        # 给定α,β,tk
        alpha = 0.2
        beta = 0.5
        tk = 1
        # 4.回溯直线搜索
        while( fun(x+tk*dk) > fun(x)-alpha*tk*lambda_k2) :
            tk = beta*tk

        # 5. 新的数据
        x = x + tk*dk
        hessian = calc_hessian(x)
        grad = calc_grad(x)
        dk, w = calc_dk(hessian, A, grad, p)
        lambda_k2 = calc_lambda_k2(dk, hessian)

     print('有可行初始点的标准牛顿方法:')
     print('x=', x)
     print('最优解为', fun(x))





