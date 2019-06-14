#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: question2.py
@time: 2019/06/08
@contact: 1284975112@qq.com
@site:
@software: PyCharm

"""
import numpy as np

# 不可行初始解

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
    return np.log(x) + 1

def calc_hessian(x):
    len = x.size
    result = np.zeros((len, len))
    for i in range(len):
        result[i][i] = 1/x[i]
    return result

def calc_dk(hessian, A, grad, b, v, p):
    len = grad.size
    zero1 = np.zeros((p, p))


    # 进行拼接组合
    A_col1 = np.concatenate((hessian, A), axis=0)
    A_t = A.T
    A_col2 = np.concatenate((A_t, zero1), axis=0)
    A_new = np.concatenate((A_col1, A_col2), axis=1)

    new_b_rol1 = grad + np.dot(A_t, v)
    new_b_rol2 = np.dot(A, x) - b
    b_new = -np.concatenate((new_b_rol1, new_b_rol2), axis=0)

    # 计算
    x_new = np.linalg.solve(A_new, b_new)
    dk = x_new[:len]
    dv = x_new[len:]
    return dk, dv

def calc_lambda_k2(dk, hessian):
    dk_t = dk.T
    lambda_k2 = np.dot(np.dot(dk, hessian), dk_t)
    return lambda_k2

def calc_r_matrix(x, v, A, b):
    A_t = A.T
    rol1 = calc_grad(x) + np.dot(A_t, v)
    rol2 = np.dot(A, x) - b
    r_matrix =  np.concatenate((rol1, rol2), axis=0)
    return r_matrix

def calc_norm2(mat):
    return np.sqrt(np.sum(mat**2))


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
     v = np.ones(p)
     error_value = 1e-10

     # 2.确定dk,dv
     # 进行矩阵拼接

     hessian = calc_hessian(x)
     grad = calc_grad(x)

     dk,dv = calc_dk(hessian, A, grad, b, v, p)

     # 3.停止准则 等式条件怎么搞
     while calc_norm2(calc_r_matrix(x, v, A, b)) > error_value :
        # 给定α,β,tk
        alpha = 0.2
        beta = 0.5
        tk = 1
        # 4.回溯直线搜索
        while calc_norm2(calc_r_matrix(x+tk*dk, v+tk*dv, A, b)) > (1-alpha*tk)*calc_norm2(calc_r_matrix(x, v, A, b)) :
            tk = beta*tk
        # 5. 新的数据
        x = x + tk*dk
        v = v + tk*dv
        hessian = calc_hessian(x)
        grad = calc_grad(x)
        dk, dv = calc_dk(hessian, A, grad, b, v, p)

     print('不可行初始点的标准牛顿方法:')
     print('v=', v)
     print('x=',x)
     print('最优解为',fun(x))





