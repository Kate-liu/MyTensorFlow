# TensorFlow Basic Concept

## Install package

- pip install tensorflow==1.14.0



## TensorFlow modules and APIs
- High-level
    - Estimators
    - keras
- Mid-level
    - Layers
    - Datasets
    - Metrics
- Low-level
    - Python
    - c++
    - java
    - go
- kernel
    - Tensorflow Distributeed Execution Engine

![](./Data/TensorFlow%20modules%20and%20APIs.PNG)



## TensorFlow Architecture

- API
- Distribute
- Common
- kernel function
- XLA
- Net
- hardware


![](./Data/TensorFlow%20Architecture.PNG)



## TensorFlow Data flow diagram
- 声明式编程
    - 核心思想是要什么
    - 程序抽象为数学模型
    - ...
    
- 命令式编程
    - 核心思想是怎么做
    - 程序抽象为有穷自动机
    - ...

![](./Data/声明式编程与命令式编程的多角度对比.PNG)
    
    
- 数据流图
    - 有向边
        - 张量Tensor
        - 稀疏张量SparseTensor
        
    - 节点
        - 计算节点Operation
        - 存储结点Variable
        - 数据结点Placeholder

![](./Data/TensorFlow数据流图.PNG)



  
- 数据流图优势
    - 并行计算快
    - 分布式计算快
    - 预编译优化(XLA)
    - 可移植性好(Language-independent representation)
    - 人工与真实数据均变现良好




## Tensor
- 在数学中，张量是一种几何实体，广义上表示任意形式的数据
- 在tensorflow中，张量表示某种相同数据类型的多维数组

- TensorFlow张量是什么
    - 张量是用来表示多维数据的
    - 张量是执行操作时的输入或输出数据
    - 用户通过执行操作来创建或计算张量
    - 张量的形状不一定在编译时确定，可以在运行时通过形状推断计算得出


![](./Data/张量Tensor.PNG)



- 张量的创建
    - 常量
        - tf.constant
    - 占位符
        - tf.placeholder
    - 变量
        - tf.Variable


## Variable

- tensorflow变量的主要作用是维护特定节点的状态，如深度学习或机器学习的模型参数
- tf.Variable方法是操作，返回值是变量（特殊张量）

- 变量使用流程
    - tf.Variable
    - tf.train.Saver

![](./Data/tensorflow变量使用流程.PNG)

- Saver使用示例
```PYTHON
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')
# 指定需要保存和恢复的变量
saver = tf.train.Saver({'v1': v1, 'v2': v2})
saver = tf.train.Saver([v1, v2])
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
# 保存变量的方法
tf.train.saver.save(sess, 'my-model', global_step=0) # ==> filename: 'my-model-0'
```




