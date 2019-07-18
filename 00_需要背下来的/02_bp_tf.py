#coding:utf-8
import tensorflow as tf
import numpy as np

# batch_size：一次喂给神经网络多少数据（此数值不可以过大，否则会吃不消）
BATCH_SIZE = 8
seed = 23455

# 基于随机种子23455生成数据集
rng = np.random.RandomState(seed)
# 生成32行数据，每组数据都有 体积和重量 两个属性作为特征
X = rng.rand(32,2)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print("X:\n",X)
print("Y:\n",Y)

# 搭建NN的输入、输出、参数（其中y是矩阵计算后的值，而y_是从矩阵Y中取出来的标签
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 搭建NN的前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 均方误差计算损失
loss = tf.reduce_mean(tf.square(y-y_))

# 梯度下降开始学习，学习率0.001
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # 输出训练前的权重
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")
    
    # 训练模型
    STEPS = 12001
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        # 分批喂入训练数据，进行权重的学习（梯度下降法）
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
#         print("train_step\n",train_step)
        
        # 输出训练后的权值参数
#         print("\n")
#         print("w1:\n",sess.run(w1))
#         print("w2:\n",sess.run(w2))
        
        # 每500次计算一次均方误差
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
            print("After %d training steps, loss_mse on all data is %g" % (i+1, total_loss))