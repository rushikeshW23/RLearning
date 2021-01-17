import random
import numpy as np
import gym
import time
from gym_minigrid.wrappers import *

def sarsa(q):

    alpha = 0.7
    gamma = 0.7
    
    for z in range(300):
    
        i = env.agent_pos[0]
        j = env.agent_pos[1]
        k = env.agent_dir
        done = False
        eps = (50/(50+z))

        steps = 0

        while(steps < 200 and done == False):
            eps = (50/(50+z))

            action = actionCal(q , i , j , k ,eps)
            obs, reward, done, info = env.step(action)

            action2 = actionCal(q , i , j , k ,eps)
        
            env.render()
            p = env.agent_pos[0]
            y = env.agent_pos[1]
            r = env.agent_dir

        
            q[i][j][k][action] = q[i][j][k][action] + alpha*(reward + gamma*q[p][y][r][action2] - q[i][j][k][action])
            
            i = p
            j = y
            k = r

            steps += 1
            #time.sleep(0.15)
            if done == True:
                env.reset()
                print(q)
                print(done)
                print("steps = ",steps)
                #time.sleep(0.25)
            
        print("episodes = ",z)
        env.reset()
        time.sleep(0.25)



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

    q = np.zeros([8,8,4,3])

    sarsa(q)