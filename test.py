# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 13:11
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : test.py
# @Software: PyCharm


import os
from CGAN.CGAN import CGAN
from Config.Config import MnistConfig

def run_main():
    """
    这是主函数
    """
    weight_path = os.path.abspath("./model/20191009134644/Epoch1378dcgan_loss1.5952800512313843discriminator_loss[0.49839333 0.7379193 ]acc[0.49839333 0.7379193 ].h5")
    result_path = os.path.abspath("./test_result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    cfg =  MnistConfig()
    cgan = CGAN(cfg,weight_path)
    cgan.save_image(0,result_path)


if __name__ == '__main__':
    run_main()