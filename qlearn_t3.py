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
    
class QTableTrainer(object):
    """ class for encapsulating learning the game of tic tac toe """
    def __init__(self, lr=0.8, gamma=0.95):
        """ Creates a trainer that uses a QTable to learn how to play tic tac toe

            args: lr    - learning rate
                  gamma - maximum discounted future reward
        """

        self.lr = lr
        self.gamma = gamma
        self.Q = np.zeros([env.state_space, env.action_space])        
        
        
    def learn(self, env, num_episodes):
        """ learn the game using the provided enviromnent for the given number of episodes
            
            args: env - tic tac toe environment
                  num_episodes - number of iterations to learn from
                  
            returns: reward - the average cumulative reward
        """
        rList = []
        
        #%% runn
        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            # cumulateive reward
            cumulative_reward = 0
            done = False
            j = 0
            #The Q-Table learning algorithm
            while j < 5:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space)*(1./(i+1)))
                a = pick_best_available_action(self.Q, s, env, i)
                #Get new state and reward from environment
                s1, reward, done = env.playX(a)
                #Update Q-Table with new knowledge
                self.Q[s,a] = self.Q[s,a] + self.lr*(reward + self.gamma*np.max(self.Q[s1,:]) - self.Q[s,a])
                cumulative_reward += reward
                s = s1
                if done == True:
                    break
                # now play random play for O
                _, _, done = env.playO(random_choice(env))
                if done == True:
                    break
            # end while
            rList.append(cumulative_reward)
        # end for    
        return sum(rList)/num_episodes

    def reset(self):
        self.Q.fill(0)
    
############################
# Start Script
###########################
# the tic tac toe environment
env = t3.make()
# 
trainer = QTableTrainer()
average_cumulative_reward = trainer.learn(env, 2000);
#%% Results
print("Score over time: " +  str(average_cumulative_reward))


