from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *
from loader import Loader
from test import *

#_/_/_/ paths _/_/_/ 
DATA_PATH          = "/mnt/hd1/puwenbo/Dataset/registration2D_dataset/new"
SAVE_PATH            = "./model/pixel-reg"
 
#_/_/_/ training parameters _/_/_/
TEST_NUM = 50
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
N_EPISODES           = 3000
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 300
TEST_EPISODES = 300
GAMMA = 0.95 # discount factor

DATA_SIZE = 64

#noise setting
# MEAN = 0
# SIGMA = 15

N_ACTIONS = 10
MOVE_RANGE = 3
GPU_ID = 2

def test(loader, agent, fout):
    sum_reward = 0
    moving, fixed = loader.load_data(-1)
    current_state = State.State()
    current_state.reset(moving, fixed)
    del moving, fixed
    for t in range(0, EPISODE_LEN):
        previous_image = current_state.warp_image.copy()
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        del action, inner_state
        reward = np.square(current_state.fixed_image - previous_image)*255 - np.square(current_state.fixed_image - current_state.warp_image)*255
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode()
 
    print("test total reward {a}".format(a=sum_reward*255/TEST_NUM))
    fout.write("test total reward {a}\n".format(a=sum_reward*255/TEST_NUM))
    sys.stdout.flush()
 
 
def main(fout):

    loader = Loader(path=DATA_PATH, test_num=TEST_NUM, batch_size=TRAIN_BATCH_SIZE, data_size=DATA_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State()

    model = MyFcn(N_ACTIONS)

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()

    i = 0
    for episode in range(1, N_EPISODES+1):
        # display current state
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        moving, fixed = loader.load_data(i)
        # initialize the current state and reward
        current_state.reset(moving, fixed)
        del moving, fixed
        reward = np.zeros(current_state.warp_image.shape, current_state.warp_image.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.warp_image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            del action, inner_state

            reward = np.square(current_state.fixed_image - previous_image)*255 - np.square(current_state.fixed_image - current_state.warp_image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA, t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward*255))
        fout.write("train total reward {a}\n".format(a=sum_reward*255))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            test(loader, agent, fout)
        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
        if (i + 1) * TRAIN_BATCH_SIZE > len(loader.train_fixed_list):
            i = 0


        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)

    final_test(loader, agent, fout)

 
if __name__ == '__main__':

    fout = open('log.txt', "w")
    start = time.time()
    main(fout)
    end = time.time()
    print("{s}[s]".format(s=end - start))
    print("{s}[m]".format(s=(end - start)/60))
    print("{s}[h]".format(s=(end - start)/60/60))
    fout.write("{s}[s]\n".format(s=end - start))
    fout.write("{s}[m]\n".format(s=(end - start)/60))
    fout.write("{s}[h]\n".format(s=(end - start)/60/60))
    fout.close()
