import numpy as np
import math

def get_m(act, arg):
    [i, j, k] = arg
    type = act[i, j, k]
    a = j
    b = k
    # right
    if type == 1:
        a = j + 1
        b = k
    # left
    if type == 2:
        a = j - 1
        b = k
    # up
    if type == 3:
        a = j
        b = k + 1
    # down
    if type == 4:
        a = j
        b = k - 1
    # # rotate
    # if type == 5:
    #     print('rotate')
    #     print(j, k)
    #     print(a, b)
    #     a = int(j * math.cos(1 / 180 * math.pi) + k * math.sin(1 / 180 * math.pi))
    #     b = int(k * math.cos(1 / 180 * math.pi) - j * math.sin(1 / 180 * math.pi))
    # # -rotate
    # if type == 6:
    #     a = int(j * math.cos(-1 / 180 * math.pi) + k * math.sin(-1 / 180 * math.pi))
    #     b = int(k * math.cos(-1 / 180 * math.pi) - j * math.sin(-1 / 180 * math.pi))
    return a, b

class State():
    def __init__(self):
        self.warp_image = None
        self.fixed_image = None
    
    def reset(self, w, f):
        self.warp_image = w
        self.fixed_image = f
        size = self.warp_image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.warp_image - self.fixed_image, prev_state), axis=1)

    # def set(self, m):
    #     self.warp_image = m
    #     # self.fixed = f
    #     self.tensor[:,:self.warp_image.shape[1],:,:] = self.warp_image

    def step(self, act, inner_state):

        tmp_img = np.zeros(self.warp_image.shape, dtype=np.float32)
        b, c, h, w = self.warp_image.shape
        for i in range(0, b):
            # point
            for j in range(0, h):
                for k in range(0, w):
                    # if self.warp_image[i, 0, j, k] > 0:
                    a, b = get_m(act, [i, j, k])
                    a = min(a, w-1)
                    a = max(0, a)
                    b = min(b, w-1)
                    b= max(0, b)
                    if act[i, j, k] == 5:
                        tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] + 1.0/255
                    if act[i, j, k] == 6:
                        tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] - 1.0/255
                    if act[i, j, k] == 7:
                        tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] + 10.0/255
                    if act[i, j, k] == 8:
                        tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] - 10.0/255
                    else:
                        tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k]
            # todo interpo


        self.warp_image = tmp_img
        self.tensor[:,:self.warp_image.shape[1],:,:] = self.warp_image - self.fixed_image
        self.tensor[:,-64:,:,:] = inner_state