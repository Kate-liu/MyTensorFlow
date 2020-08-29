# -*- coding: utf-8 -*-


from keras.datasets import mnist
import matplotlib.pyplot as plt

# pip install keras==2.2.4
# pip install tensorflow==1.13.1
# the arguments path is relative to ~/.keras/datasets
# origin='https://s3.amazonaws.com/img-datasets/mnist.npz'
# Returns
# Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

############################################################################################
fig = plt.figure()
for i in range(15):
    plt.subplot(3, 5, i + 1)  # 绘制前15个手写体数字，以3行5列子图形式展示
    plt.tight_layout()  # 自动适配子图尺寸
    plt.imshow(x_train[i], cmap='Greys')  # 使用灰色显示像素灰度值
    plt.title("{}-Label: {}".format(i, y_train[i]))  # 设置标签为子图标题
    plt.xticks([])  # 删除x轴标记
    plt.yticks([])  # 删除y轴标记
plt.show()

fig = plt.figure()
for i in range(15):
    plt.subplot(3, 5, i + 1)  # 绘制前15个手写体数字，以3行5列子图形式展示
    plt.tight_layout()  # 自动适配子图尺寸
    plt.imshow(x_test[i], cmap='Greys')  # 使用灰色显示像素灰度值
    plt.title("{}-Label: {}".format(i, y_test[i]))  # 设置标签为子图标题
    plt.xticks([])  # 删除x轴标记
    plt.yticks([])  # 删除y轴标记
plt.show()

# 28 * 28 D
print(x_train[0])