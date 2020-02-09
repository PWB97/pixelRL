import cv2
import os
import numpy as np


class Loader(object):
    def __init__(self, path, test_num, batch_size, data_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.path = path
        self.fixed_list = os.listdir(path + '/MR')
        self.moving_list = os.listdir(path + '/bs_ct')

        self.train_fixed_list= self.fixed_list[:-test_num]
        self.test_fixed_list = self.fixed_list[-test_num:]

        self.train_moving_list = self.moving_list[:-test_num]
        self.test_moving_list = self.moving_list[-test_num:]

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

if __name__ == '__main__':
    DATA_PATH = "/mnt/hd1/puwenbo/Dataset/registration2D_dataset/new"
    SAVE_PATH = "./model/pixel-reg"
    TEST_NUM = 50
    TRAIN_BATCH_SIZE = 64
    DATA_SIZE = 64


    loader = Loader(path=DATA_PATH, test_num=TEST_NUM, batch_size=TRAIN_BATCH_SIZE, data_size=DATA_SIZE)
    moving, fixed = loader.load_data(-1)
    print(moving.shape)
    print(fixed.shape)

    print(moving[0].shape)

    res = np.maximum(0, moving[0][0])
    res = (res * 255).astype(np.uint8)


    cv2.imwrite('test.png', res)