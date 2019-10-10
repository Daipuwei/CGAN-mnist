# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 17:29
# @Author  : Dai PuWei
# @File    : MnistGenerator.py
# @Software: PyCharm

import math
import numpy as np
from keras.datasets import mnist

class MnistGenerator(object):

    def __init__(self,batch_size):
        """
        这是图像数据生成器的初始化函数
        :param batch_size: 小批量样本规模
        """
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        #self.x = np.concatenate([x_train,x_test]).astype(np.float32)
        self.x = np.expand_dims((x_train.astype(np.float32)-127.5)/127.5,axis=-1)
        #self.y = to_categorical(np.concatenate([y_train,y_test]),num_classes=10)
        self.y = y_train.reshape(-1,1)
        #self.y = self.y[y == ]
        #print(np.shape(self.x))
        #print(np.shape(self.y))
        self.images_size = len(self.x)
        random_index = np.random.permutation(np.arange(self.images_size))
        self.x = self.x[random_index]
        self.y = self.y[random_index]

        self.epoch = 1                                  # 当前迭代次数
        self.batch_size = int(batch_size)
        self.batch_num = math.ceil(self.images_size / self.batch_size)
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
            yield batch_images,batch_labels

    def next_batch(self):
        datagen = self._next_batch()
        return datagen.__next__()
