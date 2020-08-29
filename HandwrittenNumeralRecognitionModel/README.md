# Handwritten numeral recognition model



## 手写体数字MNIST数据集介绍
- MNIST 数据集介绍
    - http://yann.lecun.com/exdb/mnist/
    - 由纽约大学的 Yann LeCun 等人维护
    
- 获取 MNIST 数据集

- MNIST 手写体数字介绍
    - ［28， 28］ 的二阶数组来表示每个手写体数字
    -  MNIST 数据集中的图像都是256阶灰度图， 即灰度值 0 表示白色（背景） ， 255 表示黑色（前景）
    -  做数据规范化， 将灰度值缩放为［0， 1］ 的float32数据类型
    - 下载和读取 MNIST 数据集
        - from tensorflow.examples.tutorials.mnist import input_data
        - mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    - tf.contrib.learn 模块已被废弃
    
- 使用 Keras 加载 MNIST 数据集
    - tf.kera.datasets.mnist.load_data(path=‘mnist.npz’)
    
- MNIST 数据集 样例可视化

- Code
    - 使用 tf.contrib.learn 模块加载 MNIST 数据集（Deprecated）:
        - [IntroduceMNIST](./IntroduceMNIST.py)
        
    - 使用 Keras 加载 MNIST 数据集
        - [IntroduceMNISTKeras](./IntroduceMNISTKeras.py)




## MNIST Softmax 网络介绍











