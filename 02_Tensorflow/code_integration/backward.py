#coding:utf-8
#导入模块，定义反向传播，生成训练集，执行预测
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateds
import forward

# 设置训练超参数
STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

# 定义反向传播
def backward():
    # 为训练集输入占位（模拟一个变量x和y_，等搭建好后，在后面进行传入数据）
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    
    # 生成数据集
    X, Y_, Y_c = generateds.generateds()
    
    # 进行前向传播
    y = forward.forward(x, REGULARIZER)
    
    # 整体轮数
    global_step = tf.Variable(0, trainable = False)
    
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
        )
    
   # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
   #正则化步骤加入了forward中的get_weight过程，是l2正则化
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
   
   # 定义包含正则化的反向传播
    train_step = tf,train.AdamOptimizer(learning_rate).minimize(loss_total)
   
   # 定义对话，执行参数初始化，喂入数据，反向传播，预测
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict = {x:X[start:end], y_:Y_[start:end]})
            
            # 每2000轮打印一次损失值
            if i & 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict = {x:X, y_:Y_})
                print("After %d steps, loss is: %f" % (i, loss_v))
                
        # 生成网格数据，进行预测
        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        grid = np.c_[xx,ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x:grid})
        probs = probs.reshape(xx.shape)
            
    plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()
    
if __name__ == "__main__":
    backward()
            
    
