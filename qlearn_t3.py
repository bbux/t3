# -*- coding: utf-8 -*-

#%% Setup
import t3
import numpy as np


def pick_best_available_action(Q, s, env, i):
    """ pick an action with some noise from the available ones 
        
        args: Q   - the q learning matrix
              s   - the current state
              env - the tic tac toe environment
              i   - the iteration index (for scaling noise level)
              
        returns: the best choice with some noise added in
    """
    # what board moves can we chose from
    available_actions = env.available_actions()
    # current rewards forthe available actions
    q_available = [Q[s,i] for i in available_actions]
    noise = np.random.randn(1,len(q_available))*(1./(i+1))
    best_choice = np.argmax(q_available + noise)
        
    return available_actions[best_choice]

def random_choice(env):
    """ makes a random move choice from available ones
    
        args: env - the tic tac toe environment
              
        returns: a random move choice
    """
    return np.random.choice(env.available_actions())
    
    
############################
# Start Script
###########################
# the tic tac toe environment
env = t3.make()

#Initialize table with all zeros
Q = np.zeros([env.state_space, env.action_space])
# Set learning parameters
lr = .8
y = .95
num_episodes = 20
#create lists to contain total rewards and steps per episode
rList = []

#%% runn
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space)*(1./(i+1)))
        a = pick_best_available_action(Q, s, env, i)
        #Get new state and reward from environment
        s1,r,d = env.playX(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
        # now play random play for O
        env.playO(random_choice(env))

    rList.append(rAll)

#%% Results
print("Score over time: " +  str(sum(rList)/num_episodes))

print("Final Q-Table Values")
print(Q)

