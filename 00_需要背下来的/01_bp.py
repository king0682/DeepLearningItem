# -*- coding: utf-8 -*-
import numpy as np
def sigmoid(x):#激活函数
    return 1/(1+np.exp(-x))
input1 = np.array([[0.35],[0.9],[0.58],[0.78]]) #输入数据  4x1
w1 = np.random.rand(3,4)#第一层权重参数                                    3x4
w2 = np.random.rand(2,3)#第二层权重参数                                    2x3
real = np.array([[0.5],[0.7]])#标签
for s in range(100):
    output1 = sigmoid(np.dot(w1,input1))#第一层输出  3x1
    output2 = sigmoid(np.dot(w2,output1))#第二层输出,也即是最终输出    2x1
    cost = np.square(real-output2)/2#误差      2x1
    delta2=output2*(1-output2)*(real-output2)          #2x1
    delta1=output1*(1-output1)*w2.T.dot(delta2)      #3x1
    w2 = w2 + delta2.dot(output1.T)                        #2x3
    w1 = w1 + delta1.dot(input1.T)                         #3x4
    print(output1)
    print(output2)
    print(cost)


