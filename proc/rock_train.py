#!/usr/bin/env python
# coding: utf-8

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys
sys.path.insert(0,'../gym')

from support import *
from model import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default = 'sparse', 
            type=str,  help = 'Choose between encoded or sparse')
    args = parser.parse_args()
    data_type = args.data_type

model = get_model(data_type)

# start logs
from easy_tf_log import tflog
from datetime import datetime
from keras import callbacks
import os

# initialize variables
log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# add a callback tensorboard object to visualize learning
tbCallBack = callbacks.TensorBoard(log_dir = log_dir, 
        histogram_freq = 0,write_graph = True, write_images = True)

import numpy as np
import gym

# gym initialization
from environment import SIMULATOR
my_sim = SIMULATOR() 
state_temp = my_sim.reset()
observation = my_sim.state_to_tensor(state_temp)
prev_input = None

# Hyperparameters to calculate discount rewards
gamma = 0.99

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0
resume = True
running_reward = None
EPOCHS_BEFORE_SAVING = 50 
moves_count = 0
MAX_MOVES = 300
get_features, pre_proc_features = get_pre_proc_info(data_type)

# load pre-trained model if exist
old_model = get_latest_file() 
if (resume and old_model != None and os.path.isfile(old_model)):
    print("loading previous weights")
    model.load_weights(old_model)

# using seed to be able to repeat training
np.random.seed(121)

while (True):
    # preprocess the gym tensor
    cur_input = observation
    x = differ(cur_input, prev_input) if prev_input is not None else np.zeros(70)
    x = x[10:80] if prev_input is not None else x
    
    prev_input = cur_input
    x = np.array(get_features(x))
    #print_map(x)
    x_shape = x.shape
    x_t = pre_proc_features.fit_transform(x.reshape(-1, 1))
    x_t = x_t.reshape(x_shape)

    # forward the policy network and sample action according to the proba distribution
    proba = model.predict(np.expand_dims(x_t, axis=1).T)
    prob = np.random.multinomial(1, [1/4]*4, 1) 

    # used for debugging
    MODEL = 0
    RANDOM = 1

    print("Model o/p: {0}; argmax = {1}".format(proba[0].tolist(), proba[0].argmax())) 
    print("Exploration o/p: {0}; argmax = {1}".format(prob[0].tolist(), prob[0].argmax()))
    select = MODEL if (np.random.uniform() < proba[0][proba[0].argmax()])  else RANDOM
    action = proba[0].argmax() if select == MODEL else prob[0].argmax()

    # categorical action labels
    y_data = [0.0] * 4
    y_data[action] = 1.0

    x_train.append(x_t)
    y_train.append(y_data)

    print("Action = {0}; model = {1}; ".format(action, "DL" if select == MODEL else "RND"))
    
    # do one step in our environment
    state_temp, reward, done, _, _ = my_sim.step(action)
    observation = my_sim.state_to_tensor(state_temp)
    rewards.append(float(reward))
    reward_sum += float(reward)
    moves_count += 1
    if(moves_count == MAX_MOVES):
        done = True
        moves_count = 0

    # end of an episode
    if done:
        if(reward_sum > 0):
            print('At the end of episode {1} the total reward was : {1}'.format(episode_nb, reward_sum))
        
        # increment episode number
        episode_nb += 1
        
        # discount rewards & preprocessing
        from rock_karpathy import discount_rewards
        s = discount_rewards(rewards, gamma)
        from sklearn.preprocessing import StandardScaler
        pre_proc_disc = StandardScaler()
        s_shape = s.shape
        s_t = pre_proc_disc.fit_transform(s.reshape(-1, 1))
        s_t = s_t.reshape(s_shape)
        print("Discount rewards = {0}".format(s_t.tolist()))

        # training
        model.fit(x = np.vstack(x_train), 
                y = np.vstack(y_train), 
                callbacks = [tbCallBack], 
                verbose = 1, sample_weight = s_t)
        if episode_nb % EPOCHS_BEFORE_SAVING == 0:
            model.save_weights('rock_my_model_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
                                                             
        # reinitialization
        x_train, y_train, rewards = [],[],[]
        state_temp = my_sim.reset()
        observation = my_sim.state_to_tensor(state_temp)
    
        reward_sum = 0
        prev_input = None


