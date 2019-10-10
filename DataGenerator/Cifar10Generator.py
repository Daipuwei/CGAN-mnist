# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:34
# @Author  : Dai PuWei
# @Site    : 广州山越有限公司
# @File    : Cifar10Generator.py
# @Software: PyCharm

import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

class Cifar10Generator(object):

    def __init__(self,batch_size):
        """
        这是图像数据生成器的初始化函数
        :param batch_size: 小批量样本规模
        """
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        self.x = np.concatenate([x_train,x_test])[:6000]
        self.y = to_categorical(np.concatenate([y_train,y_test]))[:6000]
        #self.y = self.y[y == ]
        print(np.shape(self.x))
        print(np.shape(self.y))
        self.images_size = len(self.x)
        random_index = np.random.permutation(np.arange(self.images_size))
        self.x = self.x[random_index]
        self.y = self.y[random_index]

        self.epoch = 1                                  # 当前迭代次数
        self.batch_size = int(batch_size)
        if int(self.images_size % self.batch_size) == 0:
            self.batch_num = int(self.images_size/self.batch_size)
        else:
            self.batch_num = int(round(self.images_size / self.batch_size))
        self.start = 0
        self.end = 0
        self.finish_flag = False                        # 数据集是否遍历完一次标志

    def _next_batch(self):
        """
        :return:
        """
        while True:
            #batch_images = np.array([])
            #batch_labels = np.array([])
            if self.finish_flag:  # 数据集遍历完一次
                random_index = np.random.permutation(np.arange(self.images_size))
                self.x = self.x[random_index]
                self.y = self.y[random_index]
                self.finish_flag = False
                self.epoch += 1
            self.end = int(np.min([self.images_size,self.start+self.batch_size]))
            batch_images = self.x[self.start:self.end]
            batch_labels = self.y[self.start:self.end]
            batch_size = self.end - self.start
            if self.end == self.images_size:            # 数据集刚分均分
                self.finish_flag = True
            if batch_size < self.batch_size:        # 小批次规模小于与预定规模,基本上是最后一组
                random_index = np.random.permutation(np.arange(self.images_size))
                self.x = self.x[random_index]
                self.y = self.y[random_index]
                batch_images = np.concatenate((batch_images, self.x[0:self.batch_size - batch_size]))
                batch_labels = np.concatenate((batch_labels, self.y[0:self.batch_size - batch_size]))
                self.start = self.batch_size - batch_size
                self.epoch += 1
            else:
                self.start = self.end
            batch_images = (batch_images - 127.5)/127.5
            yield batch_images,batch_labels

    def next_batch(self):
        datagen = self._next_batch()
        return datagen.__next__()