import cv2
import os
import numpy as np
from chainer.datasets import mnist
import random


class Loader(object):
    def __init__(self, path, test_num, batch_size, data_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.path = path
        self.test_num = test_num
        self.num = None
        # self.fixed_list = os.listdir(path + '/MR')
        # self.moving_list = os.listdir(path + '/bs_ct')
        #
        # self.train_fixed_list= self.fixed_list[:-test_num]
        # self.test_fixed_list = self.fixed_list[-test_num:]
        #
        # self.train_moving_list = self.moving_list[:-test_num]
        # self.test_moving_list = self.moving_list[-test_num:]

    def load_data(self, batch_iter):
        fixed = []
        moving = []
        if batch_iter != -1:
            start = batch_iter * self.batch_size
            end = start + self.batch_size
            fixed_list_batch = self.train_fixed_list[start:end]
            for i, file in enumerate(fixed_list_batch):
                fixed.append(self.load_image(self.path + '/MR/' + file))
                moving.append(self.load_image(self.path + '/bs_ct/' + self.train_moving_list[i]))
        else:
            fixed_list_batch = self.test_fixed_list
            for i, file in enumerate(fixed_list_batch):
                fixed.append(self.load_image(self.path + '/MR/' + file))
                moving.append(self.load_image(self.path + '/bs_ct/' + self.test_moving_list[i]))

        moving_data = np.array(moving)
        fixed_data = np.array(fixed)

        return moving_data[:,np.newaxis,:,:], fixed_data[:,np.newaxis,:,:]

    def load_image(self, path):
        image = cv2.imread(path, 0)
        image = cv2.resize(image, (self.data_size, self.data_size))
        return (image / 255).astype(np.float32)


    def load_mnist_full(self):
        m_train, _ = mnist.get_mnist(withlabel=True, ndim=1)
        data_0 = []
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        data_5 = []
        data_6 = []
        data_7 = []
        data_8 = []
        data_9 = []

        fixed = []
        moving = []

        for x, t in m_train:
            x = x.reshape(28, 28)
            if 0 == t:
                data_0.append(x)
            if 1 == t:
                data_1.append(x)
            if 2 == t:
                data_2.append(x)
            if 3 == t:
                data_3.append(x)
            if 4 == t:
                data_4.append(x)
            if 5 == t:
                data_5.append(x)
            if 6 == t:
                data_6.append(x)
            if 7 == t:
                data_7.append(x)
            if 8 == t:
                data_8.append(x)
            if 9 == t:
                data_9.append(x)
        for data_n in [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7,data_8, data_9]:
            data = random.sample(data_n, 2)
            fixed.append(data[0])
            moving.append(data[1])

        moving_data = np.array(moving)
        fixed_data = np.array(fixed)

        return moving_data[:, np.newaxis, :, :], fixed_data[:, np.newaxis, :, :]



    def load_mnist(self, num, batch_iter):
        m_train, _ = mnist.get_mnist(withlabel=True, ndim=1)
        data = []
        for x, t in m_train:
            if t == num:
                x = x.reshape(self.data_size, self.data_size)
                data.append(x)

        self.num = len(data)
        moving_data = np.array(data[2*batch_iter*self.batch_size:2*batch_iter*self.batch_size+self.batch_size])
        fixed_data = np.array(data[2*batch_iter*self.batch_size+self.batch_size:2*(batch_iter*self.batch_size+self.batch_size)])

        return moving_data[:, np.newaxis, :, :], fixed_data[:, np.newaxis, :, :]


    def load_mnist_test(self, num):
        _, m_test = mnist.get_mnist(withlabel=True, ndim=1)
        data = []
        for x, t in m_test:
            if t == num:
                x = x.reshape(self.data_size, self.data_size)
                data.append(x)
        moving_data = np.array(
            data[0:self.test_num])
        fixed_data = np.array(data[self.test_num:2*self.test_num])

        return moving_data[:, np.newaxis, :, :], fixed_data[:, np.newaxis, :, :]





if __name__ == '__main__':
    DATA_PATH = "/mnt/hd1/puwenbo/Dataset/registration2D_dataset/new"
    SAVE_PATH = "./model/pixel-reg"
    TEST_NUM = 50
    TRAIN_BATCH_SIZE = 64
    DATA_SIZE = 28


    loader = Loader(path=DATA_PATH, test_num=TEST_NUM, batch_size=TRAIN_BATCH_SIZE, data_size=DATA_SIZE)
    moving, fixed = loader.load_mnist_full()
    # moving, fixed = loader.load_data(-1)
    # print(moving.shape)
    # print(fixed.shape)
        #
        # print(moving[0].shape)
        #
        # res = np.maximum(0, moving[0][0])
        # res = (res * 255).astype(np.uint8)
    #     #
    #     #
    #     # cv2.imwrite('test.png', res)

    # moving, fixed = loader.load_mnist(9,1)
    # print(loader.num)
    # print(moving[0])
    # print(moving.shape)
    #     # # print(fixed.shape)
    #     # # res = np.maximum(0, moving[0][0])
    #     # # res = (res * 255).astype(np.uint8)
    #     # # cv2.imwrite('fixed1.png', res)
    #     # image = cv2.imread('fixed1.png', 0)
    #     # print(np.shape(image))

    for i in range(10):
        warp = np.maximum(0, fixed[i][0])
        warp = (warp * 255).astype(np.uint8)
        cv2.imwrite(str(i)+'_output.png', warp)
        warp = np.maximum(0, moving[i][0])
        warp = (warp* 255).astype(np.uint8)
        cv2.imwrite(str(i)+'_input.png', warp)

    # if not os.path.exists('./' + str(1)):
    #     os.mkdir('./' + str(1))