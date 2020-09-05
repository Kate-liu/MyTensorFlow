# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha
import random
import tensorflow.gfile as gfile

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


def create_captcha_dataset(size=100,
                           data_dir='./data/',
                           height=60,
                           width=160,
                           image_format='.png'):
    # 如果保存验证码图像，先清空 data_dir 目录
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)

    # 创建 ImageCaptcha 实例 captcha
    captcha = ImageCaptcha(width=width, height=height)

    for _ in range(size):
        # 生成随机的验证码字符
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        captcha.write(text, data_dir + text + image_format)

    return None


if __name__ == '__main__':
    # 创建并保存训练集
    create_captcha_dataset(size=TRAIN_DATASET_SIZE, data_dir=TRAIN_DATA_DIR)
    # 创建并保存测试集
    create_captcha_dataset(size=TEST_DATASET_SIZE, data_dir=TEST_DATA_DIR)
