# encoding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd


# ps: if error, then pip install tensorflow==1.13.1

def normalize(data):
    return data.apply(lambda column: (column - column.mean()) / column.std())


# read
data = pd.read_csv("Data/data1.csv", names=["square", "bedroom", "price"])

# normalize
data_normalize = normalize(data)

# get ones
ones = pd.DataFrame({'ones': np.ones(len(data_normalize))})

# add ones
# axis : {0/'index', 1/'columns'}, default 0
ones_data_normalize = pd.concat([ones, data_normalize], axis=1)

# get X Y data
x_data = np.array(ones_data_normalize[ones_data_normalize.columns[0:3]])
y_data = np.array(ones_data_normalize[ones_data_normalize.columns[-1]]).reshape(len(ones_data_normalize), 1)

# ============================================================================================================
# use tensorflow
alpha = 0.01  # 学习率 alpha
epoch = 500  # 训练全量数据集的轮数

with tf.name_scope("input"):
    # 输入 X，形状[47, 3]
    x = tf.placeholder(tf.float32, x_data.shape, name="x")
    # 输出 y，形状[47, 1]
    y = tf.placeholder(tf.float32, y_data.shape, name="y")

with tf.name_scope("hypothesis"):
    # 权重变量 W，形状[3,1]
    W = tf.get_variable("weights", (x_data.shape[1], 1), initializer=tf.constant_initializer())

    # 假设函数 h(x) = w0*x0 + w1*x1 + w2*x2, 其中x0恒为1
    # 推理值 y_predication  形状[47,1]
    y_predication = tf.matmul(x, W, name="y_predication")

with tf.name_scope("loss"):
    # 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
    # tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
    # 损失函数操作 loss
    loss_op = 1 / (2 * len(x_data)) * tf.matmul((y_predication - y), (y_predication - y), transpose_a=True)

with tf.name_scope("train"):
    # 随机梯度下降优化器 opt
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # 单轮训练操作 train_op
    train_op = opt.minimize(loss_op)

# ============================================================================================================
# create session
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())

    # 创建FileWriter实例，并传入当前会话加载的数据流图
    writer = tf.summary.FileWriter("./summary/linear", sess.graph)

    # 开始训练模型，
    # 因为训练集较小，所以每轮都使用全量数据训练，如果训练量大，采用批梯度下降算法
    for e in range(1, epoch + 1):
        sess.run(train_op, feed_dict={x: x_data, y: y_data})

        if e % 10 == 0:
            loss, w = sess.run([loss_op, W], feed_dict={x: x_data, y: y_data})
            log_str = "Epoch %d \t Loss=%.4g \t Model: y=%.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))

writer.close()
