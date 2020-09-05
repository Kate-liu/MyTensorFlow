# Verification code recognition model

## 开发环境
- pip install Pillow captcha pydot flask
- Pillow (PIL Fork)
    - PIL(Python Imaging Library) 为 Python 解释器添加了图像处理功能
    - Pillow 是由 Alex Clark 及社区贡献者 一起开发和维护的一款分叉自 PIL 的图像工具库。
- captcha
    - Captcha 是一个生成图像和音频验证码的开源工具库。
- pydot
    - pydot 是用纯 Python 实现的 GraphViz 接口， 支持使用 GraphViz 解析和存储 DOT语言（graph description language） 
- flask
    - flask 是一个基于 Werkzeug 和 jinja2 开发的 Python Web 应用程序框架



## 生成验证码数据集
- 验证码（CAPTCHA） 简介
    - 全自动区分计算机和人类的公开图灵测试（英语： Completely Automated Public Turing testto tell Computers and Humans Apart， 简称CAPTCHA） ， 俗称验证码， 是一种区分用户是计算机或人的公共全自动程序。
- 验证码（CAPTCHA） 破解
- 验证码（CAPTCHA） 演进
- 验证码（CAPTCHA） 生成
    - 使用 Pillow（PIL Fork） 和 captcha 库生成验证码图像

- Code:
    - [CreateCaptchaDatasetSaveToDisk](CreateCaptchaDatasetSaveToDisk.py)
    
    - [CreateCaptchaDatasetSaveToMemory](CreateCaptchaDatasetSaveToMemory.py)


## 











