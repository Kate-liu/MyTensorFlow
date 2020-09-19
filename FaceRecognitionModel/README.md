# Face recognition model


## 人脸识别问题概述
- 人脸识别概述
    - 人脸识别，特指利用分析比较人脸视觉特征信息，进行身份鉴别的计算机技术
    - 人脸识别，属于生物特征识别技术，是对生物本身的生物特征来区分生物体个体
    - 生物特征识别技术，包括脸，指纹，手掌纹，虹膜，视网膜，声音，语音，体形，个人习惯，敲击键盘的力度和频率，签字
    - 识别技术，包括人脸识别，指纹识别，掌纹识别，虹膜识别，视网膜识别，语音识别，体形识别，键盘敲击识别，签字识别
    
- 人脸识别的技术困难
    - 不同个体之间的区别不大，所有的人脸的结构都相似
    - 人脸的外形很不稳定，人可以通过脸部变化，产生很多表情
    - 人脸识别，受光照条件，遮盖物，年龄，姿态角度等影响
    
- 人脸识别典型流程
    - 三步走，人脸检测，人脸对其，人脸特征表示
    - 传统机器学习的人脸识别，分为高维人工特征提取，降维两个步骤
    - 深度学习，从原始图像空间，直接学习判别性的人脸表示，实现端到端的人脸识别模型
    
- 深度学习引爆人脸识别
    - Google研究人员，2015年在CVPR上发表一篇开创性论文：FaceNet
    - FaceNet是一个解决人脸识别和人脸聚类问题的全新深度神经网络架构
    
- 人脸识别应用
    - 安防
    - 安检
    - 个人相册管理
    - 支付
    - KYC
    - ...
    
    
    
    
## 典型人脸相关数据集介绍
- 人脸识别数据集
    - LFW(Labeled Face in the world)
    - Youtube Faces DB
    - CASIA-WebFace
    - ...
    
- 人脸检测数据集
    - FDDB: FACE Detection Data Set and Benchmark
    - WIDER FACE： A Face Detection Benchmark
    - Large-scale CelebFaces Attributes (Celeb A) Dataset
    - ....




## 人脸识别算法介绍
- 人脸识别算法流程
    - 人脸检测
        - Face detection
    - 人脸对齐
        - Face Alignment
    - 人脸特征表征
        - Feature Representation
        
- 人脸识别-研究进展
    - 早期算法：基于几何特征，模板匹配，子空间
    - 人工特征+分类器
    - 基于深度学习的算法
    
- 早期算法-线性降维
    - 使用PCA降维得到特征脸Eigenface
    - LDA降维的大Fisherface实现人脸识别
    
- 早期算法-非线性降维
    - 流形学习是一种非线性降维方法
    
- 人工特征+分类器
    - HOG,SIFT,Gabor，LBP
    - LBP，局部二值模式特征，解决了光照敏感问题
    - 联合贝叶斯是对贝叶斯人脸的改进方法，选用LBP和LE作为基础特征
    - MSRA"Feature Master"
        - 使用高维度特征在人脸验证中，以LBP为例
        
- 基于深度学习的人脸识别
    - Facebook DeepFace
        - 使用3D模型解决人脸对齐问题
        - 使用9层深度神经网络来做人脸特征表示
        - 损失函数使用Softmax Loss
        - 通过特征嵌入Feature Embedding得到固定长度的人脸特征向量
    - Google FaceNet
        - 使用三元组锁时函数Triplet Loss
        - 得到一个紧凑的128维人脸特征




## 人脸检测工具介绍
- AI开放平台

- Opencv
    - 开源计算机视觉库，Open Source Computer Vision Library
    - pip install opencv-python
    - 使用Opencv进行人脸检测
        - [FaceDetectOpencv](./FaceDetectOpencv.py)
    
- face_recognition
    - pip insatll face_recognition
    - 使用face_recognition进行人脸检测
        - [FaceDetectFaceRecognition](./FaceDetectFaceRecognition.py)














