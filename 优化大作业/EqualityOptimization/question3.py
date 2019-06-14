#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: question3.py
@time: 2019/06/08
@contact: 1284975112@qq.com
@site:
@software: PyCharm

"""
import numpy as np

# 求解对偶问题

# 产生0,10之间的随机数
def generateA(p, n):
    return np.random.rand(p, n)*10

# 产生0,1的x_hat
def generate_x_hat(n):
    return np.random.rand(n, 1)


def fun(x, b, A):
    result = -np.dot(b,x)
    n = x.size
    for i in range(100):
        result = result - np.exp(-np.dot(x,A[:,i])-1)
    return -result

def calc_grad(x, b, A):
    n = x.size
    result = np.zeros(n)
    for i in range(n):
        sum = 0
        for k in range(100):
            sum = sum + A[i][k]*np.exp(-np.dot(x,A[:,k])-1)
        result[i] = -b[i] + sum
    return -result

def calc_hessian(x, A):
    n = x.size
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            num = 0
            for k in range(100):
                num =  num + A[i][k]*A[j][k]*np.exp(-np.dot(x,A[:,k])-1)
            result[i][j] = -num
    return -result

def calc_dk(hessian, grad):
    dk = -np.dot(np.linalg.inv(hessian), grad)
    return dk

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
     x = np.ones(p)*0.1
     hessian = calc_hessian(x, A)
     grad = calc_grad(x, b, A)

     dk = calc_dk(hessian, grad)
     lambda_k2 = calc_lambda_k2(dk, hessian)
     #
     while lambda_k2/2 > error_value :
        # 给定α,β,tk
        alpha = 0.2
        beta = 0.5
        tk = 1
        # 4.回溯直线搜索
        while( fun(x+tk*dk,b,A) > fun(x,b,A)-alpha*tk*lambda_k2) :
            tk = beta*tk
        # 5. 新的数据
        x = x + tk*dk
        hessian = calc_hessian(x,A)
        grad = calc_grad(x,b,A)
        dk = calc_dk(hessian, grad)
        lambda_k2 = calc_lambda_k2(dk, hessian)
        print(fun(x, b, A))

     print('对偶牛顿方法:')# 这里的x为v('v=',x)
     print('最优解为',-fun(x,b,A))



