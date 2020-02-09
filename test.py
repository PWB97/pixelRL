from loader import Loader
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

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = "../testing.txt"
IMAGE_DIR_PATH              = "../"

DATA_PATH          = "/mnt/hd1/puwenbo/Dataset/registration2D_dataset/new"
SAVE_PATH            = "./model/pixel-reg"
DATA_SIZE = 64

#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 10
GAMMA = 0.95 # discount factor
TEST_NUM = 50
#noise setting
# MEAN = 0
# SIGMA = 15

N_ACTIONS = 10
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.

GPU_ID = 2

def final_test(loader, agent, fout):
    sum_reward = 0
    moving, fixed = loader.load_data(-1)
    current_state = State.State()
    current_state.reset(moving,fixed)

    for t in range(0, EPISODE_LEN):
        previous_image = current_state.warp_image.copy()
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        reward = np.square(current_state.fixed_image - previous_image)*255 - np.square(current_state.fixed_image - current_state.warp_image)*255
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode()

    for i in range(TEST_NUM):
        warp = np.maximum(0, current_state.warp_image[i][0])
        warp = (warp * 255).astype(np.uint8)
        moving = np.maximum(0, moving[i][0])
        moving = (moving * 255).astype(np.uint8)
        cv2.imwrite('./resultimage/'+str(i)+'_output.png', warp)
        cv2.imwrite('./resultimage/'+str(i)+'_input.png', moving)
 
    print("test total reward {a}".format(a=sum_reward*255/TEST_NUM))
    fout.write("test total reward {a}\n".format(a=sum_reward*255/TEST_NUM))
    sys.stdout.flush()
 
 
def main(fout):
    loader = Loader(path=DATA_PATH, test_num=TEST_NUM, batch_size=TRAIN_BATCH_SIZE, data_size=DATA_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('./model/pretrained_15.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    final_test(loader, agent, fout)
    
     
 
if __name__ == '__main__':
    try:
        fout = open('testlog.txt', "w")
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
    except Exception as error:
        print(error.message)
