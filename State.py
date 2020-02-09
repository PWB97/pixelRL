import numpy as np
import math

def get_m(act, arg):
    [i, j, k] = arg
    type = act[i, j, k]
    a = i
    b = j
    # right
    if type == 1:
        a = i + 1
        b = j
    # left
    if type == 2:
        a = i - 1
        b = j
    # up
    if type == 3:
        a = i
        b = j + 1
    # down
    if type == 4:
        a = i
        b = j - 1
    # rotate
    if type == 5:
        a = int(i * math.cos(1 / 180 * math.pi) + j * math.sin(1 / 180 * math.pi))
        b = int(j * math.cos(1 / 180 * math.pi) - i * math.sin(1 / 180 * math.pi))
    # -rotate
    if type == 6:
        a = int(i * math.cos(-1 / 180 * math.pi) + j * math.sin(-1 / 180 * math.pi))
        b = int(j * math.cos(-1 / 180 * math.pi) - i * math.sin(-1 / 180 * math.pi))
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
                    if self.warp_image[i, 0, j, k] > 0:
                        a, b = get_m(act, [i, j, k])
                        a = w-1 if (a > w-1) else a
                        b = w-1 if (b > w-1) else b
                        if act[i, j, k] == 7:
                            tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] + 1
                        elif act[i, j, k] == 8:
                            tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] - 1
                        elif act[i, j, k] == 9:
                            tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] + 10
                        elif act[i, j, k] == 10:
                            tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k] - 10
                        else:
                            tmp_img[i, 0, a, b] = self.warp_image[i, 0, j, k]
            # todo interpo


        self.warp_image = tmp_img
        self.tensor[:,:self.warp_image.shape[1],:,:] = self.warp_image - self.fixed_image
        self.tensor[:,-64:,:,:] = inner_state