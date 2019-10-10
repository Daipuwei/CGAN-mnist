# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:44
# @Author  : Dai PuWei
# @File    : Config.py
# @Software: PyCharm

import os

class MnistConfig(object):

    _defaults = {
        "generator_noise_input_dim": 100,
        "condational_label_num": 10,
        #"discriminator_image_input_dim": (32,32,3),
        "discriminator_image_input_dim": (28,28,1),
        "batchnormalization_momentum": 0.8,
        "dropout_prob": 0.4,
        "LeakyReLU_alpha": 0.2,
        "save_epoch_interval": 1,
        "generate_image_interval":[1,5,10,50,100,500,1000,2000,5000],
        #"is_mnist": False,
        "is_mnist": True,
    }

    def __init__(self, **kwargs):
        # 初始化相关配置参数
        self.__dict__.update(self._defaults)
        # 根据相关传入参数进行参数更新
        self.__dict__.update(kwargs)

        self.model_dir = os.path.abspath("./model")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.dataset_dir = os.path.abspath("./data")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self.train_result_dir = os.path.abspath("./train_result")
        if not os.path.exists(self.train_result_dir):
            os.mkdir(self.train_result_dir)

        self.log_dir = os.path.abspath("./logs")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
