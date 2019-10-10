# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:39
# @Author  : Dai PuWei
# @File    : CGAN.py
# @Software: PyCharm

import os
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt

from scipy.stats import truncnorm


from keras import Input
from keras import Model
from keras import Sequential

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.merge import multiply
from keras.layers.merge import concatenate
from keras.layers.merge import add
from keras.layers import Embedding
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from keras.datasets import mnist

def make_trainable(net, val):
    """ Freeze or unfreeze layers
    """
    net.trainable = val
    for l in net.layers: l.trainable = val

class CGAN(object):

    def __init__(self,config):
        """
        这是CGAN的初始化函数
        :param config: 参数配置类实例
        """
        self.config = config
        self.build_cgan_model()

    def build_cgan_model(self):
        """
        这是搭建CGAN模型的函数
        :return:
        """
        # 初始化输入
        self.generator_noise_input = Input(shape=(self.config.generator_noise_input_dim,))
        self.condational_label_input = Input(shape=(1,), dtype='int32')
        self.discriminator_image_input = Input(shape=self.config.discriminator_image_input_dim)

        # 定义优化器
        self.optimizer = Adam(lr=2e-4, beta_1=0.5)

        # 构建生成器模型与判别器模型
        self.discriminator_model = self.build_discriminator_model()
        self.discriminator_model.compile(optimizer=self.optimizer, loss=['binary_crossentropy'],metrics=['accuracy'])
        self.generator_model = self.build_generator()

        # 构建CGAN模型
        self.discriminator_model.trainable = False
        self.cgan_input = [self.generator_noise_input,self.condational_label_input]
        generator_output = self.generator_model(self.cgan_input)
        cgan_output = self.discriminator_model([generator_output,self.condational_label_input])
        self.cgan = Model(self.cgan_input,cgan_output)

        # 编译
        #self.discriminator_model.compile(optimizer=self.optimizer,loss='binary_crossentropy')
        self.cgan.compile(optimizer=self.optimizer,loss=['binary_crossentropy'])

    def build_discriminator_model(self):
        """
        这是搭建生成器模型的函数
        :return:
        """
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.config.discriminator_image_input_dim)))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.LeakyReLU_alpha))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.LeakyReLU_alpha))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.config.discriminator_image_input_dim)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.config.condational_label_num,
                                              np.prod(self.config.discriminator_image_input_dim))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)

        return Model([img, label], validity)


    def build_generator(self):
        """
        这是构建生成器网络的函数
        :return:返回生成器模型generotor_model
        """
        model = Sequential()

        model.add(Dense(256, input_dim=self.config.generator_noise_input_dim))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=self.config.LeakyReLU_alpha))
        model.add(BatchNormalization(momentum=self.config.batchnormalization_momentum))
        model.add(Dense(np.prod(self.config.discriminator_image_input_dim), activation='tanh'))
        model.add(Reshape(self.config.discriminator_image_input_dim))

        model.summary()

        noise = Input(shape=(self.config.generator_noise_input_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.config.condational_label_num, self.config.generator_noise_input_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def train(self, train_datagen, epoch, k, batch_size=256):
        """
        这是DCGAN的训练函数
        :param train_generator:训练数据生成器
        :param epoch:周期数
        :param batch_size:小批量样本规模
        :param k:训练判别器次数
        :return:
        """
        time =datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(self.config.model_dir,time)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        train_result_path = os.path.join(self.config.train_result_dir,time)
        if not os.path.exists(train_result_path):
            os.mkdir(train_result_path)

        for ep in np.arange(1, epoch+1).astype(np.int32):
            cgan_losses = []
            d_losses = []
            # 生成进度条
            length = train_datagen.batch_num
            progbar = Progbar(length)
            print('Epoch {}/{}'.format(ep, epoch))
            iter = 0
            while True:
                # 遍历一次全部数据集，那么重新来结束while循环
                #print("iter:{},{}".format(iter,train_datagen.get_epoch() != ep))
                if train_datagen.epoch != ep:
                    break

                # 获取真实图片，并构造真图对应的标签
                batch_real_images, batch_real_labels = train_datagen.next_batch()
                batch_real_num_labels = np.ones((batch_size, 1))
                #batch_real_num_labels = truncnorm.rvs(0.7, 1.2, size=(batch_size, 1))
                # 初始化随机噪声，伪造假图，并合并真图和假图数据集
                batch_noises = np.random.normal(0, 1, size = (batch_size, self.config.generator_noise_input_dim))
                d_loss = []
                for i in np.arange(k):
                    # 构造假图标签，合并真图和假图对应标签
                    batch_fake_num_labels = np.zeros((batch_size,1))
                    #batch_fake_num_labels = truncnorm.rvs(0.0, 0.3, size=(batch_size, 1))
                    batch_fake_labels = deepcopy(batch_real_labels)
                    batch_fake_images = self.generator_model.predict([batch_noises,batch_fake_labels])

                    # 训练判别器
                    real_d_loss = self.discriminator_model.train_on_batch([batch_real_images,batch_real_labels],
                                                                                      batch_real_num_labels)
                    fake_d_loss = self.discriminator_model.train_on_batch([batch_fake_images, batch_fake_labels],
                                                                                      batch_fake_num_labels)
                    d_loss.append(list(0.5*np.add(real_d_loss,fake_d_loss)))
                #print(d_loss)
                d_losses.append(list(np.average(d_loss,0)))
                #print(d_losses)

                # 生成一个batch_size的噪声来训练生成器
                #batch_num_labels = truncnorm.rvs(0.7, 1.2, size=(batch_size, 1))
                batch_num_labels = np.ones((batch_size,1))
                batch_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
                cgan_loss = self.cgan.train_on_batch([batch_noises,batch_labels], batch_num_labels)
                cgan_losses.append(cgan_loss)

                # 更新进度条
                progbar.update(iter, [('dcgan_loss', cgan_losses[iter]),
                                      ('discriminator_loss',d_losses[iter][0]),
                                      ('acc',d_losses[iter][1])])
                #print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (ep, d_losses[ep][0], 100 * d_losses[ep][1],cgan_loss))
                iter += 1
            if ep % self.config.save_epoch_interval == 0:
                model_cgan = "Epoch{}dcgan_loss{}discriminator_loss{}acc{}.h5".format(ep, np.average(cgan_losses),
                                                                                      np.average(d_losses,0)[0],np.average(d_losses,0)[1])
                self.cgan.save(os.path.join(model_path, model_cgan))
                save_dir = os.path.join(train_result_path, str("Epoch{}".format(ep)))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.save_image(int(ep), save_dir)
            '''
            if int(ep) in self.config.generate_image_interval:
                save_dir = os.path.join(train_result_path,str("Epoch{}".format(ep)))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                self.save_image(ep,save_dir)
            '''
        plt.plot(np.arange(epoch),cgan_losses,'b-','cgan-loss')
        plt.plot(np.arange(epoch), d_losses[0], 'b-', 'd-loss')
        plt.grid(True)
        plt.legend(locs="best")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(train_result_path,"loss.png"))

    def save_image(self, epoch,save_path):
        """
        这是保存生成图片的函数
        :param epoch:周期数
        :param save_path: 图片保存地址
        :return:
        """
        rows, cols = 10, 10

        fig, axs = plt.subplots(rows, cols)
        for i in range(rows):
            label = np.array([i]*rows).astype(np.int32).reshape(-1,1)
            noise = np.random.normal(0, 1, (cols, 100))
            images = self.generator_model.predict([noise,label])
            images = 127.5*images+127.5
            cnt = 0
            for j in range(cols):
                #img_path = os.path.join(save_path, str(cnt) + ".png")
                #cv2.imwrite(img_path, images[cnt])
                #axs[i, j].imshow(image.astype(np.int32)[:,:,0])
                axs[i, j].imshow(images[cnt,:, :, 0].astype(np.int32), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(save_path, "mnist-{}.png".format(epoch)), dpi=600)
        plt.close()

    def generate_image(self,label):
        """
        这是伪造一张图片的函数
        :param label:标签
        """
        noise = truncnorm.rvs(-1, 1, size=(1, self.config.generator_noise_input_dim))
        label = np.array([label]).T
        image = self.generator_model.predict([noise,label])[0]
        image = 127.5*(image+1)
        return image
