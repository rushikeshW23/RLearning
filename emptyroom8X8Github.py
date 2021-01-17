import random
import numpy as np
import gym
import time
from gym_minigrid.wrappers import *

def qfunction(qmatrix , alpha = 0.8):

    for t in range(300):

        i = env.agent_pos[0]
        j = env.agent_pos[1]
        k = env.agent_dir
        done = False
        
        steps = 0

        while(steps < 200 and done == False):
            eps = (50/(50+t))
            action = actionCal(qmatrix , i , j , k ,eps)
            env.render()
            obs, reward, done, info = env.step(action)
            env.render()
            p = env.agent_pos[0]
            q = env.agent_pos[1]
            r = env.agent_dir
            qmatrix[i][j][k][action] = qmatrix[i][j][k][action] + alpha*(reward + 0.9*np.max(qmatrix[p,q,r]) - qmatrix[i][j][k][action])
            
            i = env.agent_pos[0]
            j = env.agent_pos[1]
            k = env.agent_dir
            steps += 1
            #time.sleep(0.15)
            if done == True:
                env.reset()
                print(qmatrix)
                print(done)
                print("steps = ",steps)
                #time.sleep(0.25)
            
        print("episodes = ",t)
        env.reset()


def actionCal(qmatrix , i , j , k , eps):
    action = 0
    print("eps = ",eps)
    if np.random.uniform(0,1) < eps:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(qmatrix[i][j][k])
    return action

if __name__ == "__main__":

    env_name = 'MiniGrid-Empty-8x8-v0'
    env = gym.make(env_name)
    env = FullyObsWrapper(env)

    qmatrix = np.zeros([8,8,4,3])

    qfunction(qmatrix)