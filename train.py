# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:43
# @Author  : Dai PuWei
# @File    : train.py
# @Software: PyCharm

import os
import datetime

from CGAN.CGAN import CGAN
from Config.Config import MnistConfig
from DataGenerator.Cifar10Generator import Cifar10Generator
from DataGenerator.MnistGenerator import MnistGenerator

def run_main():
    """
    这是主函数
    """
    cfg =  MnistConfig()
    dcgan = CGAN(cfg)
    batch_size = 512
    #train_datagen = Cifar10Generator(int(batch_size/2))
    train_datagen = MnistGenerator(batch_size)
    dcgan.train(train_datagen,100000,1,batch_size)


if __name__ == '__main__':
    run_main()
