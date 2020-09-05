# -*- coding: utf-8 -*-


from captcha.image import ImageCaptcha

import random
import numpy as np

import tensorflow.gfile as gfile
import matplotlib.pyplot as plt
import PIL.Image as Image

##################################################################################
# 定义常量和字符集
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER  # 验证码字符集
CAPTCHA_LEN = 4  # 验证码长度
CAPTCHA_HEIGHT = 60  # 验证码高度
CAPTCHA_WIDTH = 160  # 验证码宽度

TRAIN_DATASET_SIZE = 5000  # 验证码数据集大小
TEST_DATASET_SIZE = 1000
TRAIN_DATA_DIR = './train-dataset/'  # 验证码数据集目录
TEST_DATA_DIR = './test-dataset/'


def gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LEN):
    text = [random.choice(charset) for _ in range(length)]
    return ''.join(text)


def gen_captcha_dataset(size=100,
                        height=60,
                        width=160):
    # 创建 ImageCaptcha 实例 captcha
    captcha = ImageCaptcha(width=width, height=height)

    # 创建图像和文本数组
    images, texts = [None] * size, [None] * size
    for i in range(size):
        # 生成随机的验证码字符
        texts[i] = gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LEN)
        # 使用 PIL.Image.open() 识别新生成的验证码图像
        # 然后，将图像转换为形如(CAPTCHA_WIDTH, CAPTCHA_HEIGHT, 3) 的 Numpy 数组
        images[i] = np.array(Image.open(captcha.generate(texts[i])))

    return images, texts


if __name__ == '__main__':
    # 生成100张验证码图像和字符
    images, texts = gen_captcha_dataset(size=100)
    # show captcha
    plt.figure()
    for i in range(20):
        plt.subplot(5, 4, i + 1)  # 绘制前20个验证码，以5行4列子图形式展示
        plt.tight_layout()  # 自动适配子图尺寸
        plt.imshow(images[i])
        plt.title("Label: {}".format(texts[i]))  # 设置标签为子图标题
        plt.xticks([])  # 删除x轴标记
        plt.yticks([])  # 删除y轴标记
    plt.show()
