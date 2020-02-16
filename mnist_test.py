from chainer.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2


train, test = mnist.get_mnist(withlabel=True, ndim=1)
x, t = train[0]

# print(x)
# plt.imshow(x.reshape(28, 28), cmap="gray")
# plt.savefig("5.png")

i = 0
for x, t in train:
    print(np.shape(x))
    if t == 9:
        i += 1



# print(i)
#
# print(t)

# print(test.shape)