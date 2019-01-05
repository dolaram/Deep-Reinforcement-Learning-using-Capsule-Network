#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import cv2
import sys
import pong_fun as game 
import random
import time 
import numpy as np
from collections import deque
from con_fun import *
actions = 6
gamma = 0.99
replay_memory = 50000
batch = 32

def trainNetwork(s, readout, h_fc1, sess):
    tick = time.time()
    a = tf.placeholder("float", [None, actions])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()
    # past 3 wins 
    win_score = []
    win_score.append(0)
    win_score.append(0)
    win_score.append(0)
    win_score.append(0)
    # store the previous observations in replay memory
    D = deque()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(actions)
    do_nothing[0] = 1
    x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    observe = 500.
    explore = 500.
    FINAL_EPSILON = 0.05
    INITIAL_EPSILON = 1.0
    epsilon = INITIAL_EPSILON
    t = 0
    K = 1
    while True:
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([actions])
        action_index = 0
        if random.random() <= epsilon or t <= observe:
            action_index = random.randrange(actions)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / explore

        for i in range(0, K):
            x_t1_col, r_t, terminal, bar1_score, bar2_score = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > replay_memory:
                D.popleft()
        
        if t > observe:
            minibatch = random.sample(D, batch)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + gamma * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        s_t = s_t1
        t += 1
        if r_t!= 0:
            print ("Timestep", t," Score", bar1_score)
        
        win_score.pop(0)
        win_score.append(bar1_score - bar2_score)
        if(np.matrix(win_score).sum() > 72): #72
            print("Game_Ends_in Time:",int(time.time() - tick))
            break;   
def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    tick = time.time()
    main()
    print("Game_Ends_in Time:",int(time.time() - tick))
    print("____________ END HERE _____________")
