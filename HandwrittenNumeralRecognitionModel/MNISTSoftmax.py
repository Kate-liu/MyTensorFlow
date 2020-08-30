# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation
import tensorflow.gfile as gfile

##################################################################################################
# 加载数据集
# x is pictures
# y is class label
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

##################################################################################################
# 数据处理：规范化
# 将图像本身从[28,28]转换为[784,]
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)

print(X_train.shape, type(X_train))
print(X_test.shape, type(X_test))

# 将数据类型转换为float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 数据归一化
X_train /= 255
X_test /= 255

##################################################################################################
# 统计训练数据中各标签数量
label, count = np.unique(y_train, return_counts=True)
print(label, "-----", count)

fig = plt.figure()
plt.bar(label, count, width=0.7, align='center')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0, 7500)

for a, b in zip(label, count):
    plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)

plt.show()

##################################################################################################
# 数据处理：One-hot编码
n_classes = 10

print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

Y_test = np_utils.to_categorical(y_test, n_classes)

print(y_train[0])
print(Y_train[0])

##################################################################################################
# 使用Keras sequential model 定义神经网络
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))  # 隐藏层

model.add(Dense(512))
model.add(Activation('relu'))  # 隐藏层

model.add(Dense(10))
model.add(Activation('softmax'))  # 输出层

##################################################################################################
# 编译模型
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')  # categorical_crossentropy：交叉熵

##################################################################################################
# 训练模型
# epochs：训练五次， batch_size：每一次输入128张图训练， validation_data：进行验证的图
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=2,
                    validation_data=(X_test, Y_test))

##################################################################################################
# 可视化指标
# loss: 0.0265 - acc: 0.9915 - val_loss: 0.0710 - val_acc: 0.9792

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

##################################################################################################
# 保存模型
save_dir = "./MNIST/model/"

if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)

model.save(model_path)
print('Saved trained model at %s ' % model_path)

##################################################################################################
# 加载模型
mnist_model = load_model(model_path)


##################################################################################################
# 统计模型在测试集上的分类结果
# Returns the loss value & metrics values for the model in test mode.
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1] * 100))

# Generate class predictions for the input samples.
predicted_classes = mnist_model.predict_classes(X_test)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print("Classified correctly count: {}".format(len(correct_indices)))
print("Classified incorrectly count: {}".format(len(incorrect_indices)))

