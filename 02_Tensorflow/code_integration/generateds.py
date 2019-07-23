#coding:utf-8
#生成数据集

import numpy as np
import tensorflow as tf
seed = 2

def generateds():
    # 基于seed生成随机数类rdm
    rdm = np.random.RandomState(seed)
    # 随机数返回300*2的矩阵，表示300组坐标点(x0,x1)，作为训练集的输入
    X = rdm.randn(300,2)
    # 将半径平方和小于2的数据标注为1，否则标注为0
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0,x1) in X]
    # 将Y_中的1标记为'red'，0标记为'blue'，方便后来画图
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集标签进行整理，X整理为2列，Y_整理成1列
    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)
    
    return X,Y_,Y_c