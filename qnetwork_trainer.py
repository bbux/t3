# -*- coding: utf-8 -*-
""" module holding the qnetwork trainer class """
import numpy as np
import tensorflow as tf

def random_choice(env):
    """ makes a random move choice from available ones

        args: env - the tic tac toe environment

        returns: a random move choice
    """
    return np.random.choice(env.available_actions())



def state_array(size, index):
    """ build array and set the index of the given state """
    array = np.zeros((1,size), dtype=np.float32)
    array[0][index] = 1.0
    return array

class QNetworkTrainer(object):
    """ class for encapsulating learning the game of tic tac toe using tensorflow
        neural network    
    """
    def __init__(self, lr=0.1, rand_choice=0.1, gamma=0.99):
        """ Creates a trainer that uses a tensorflow neural net to learn how to
            play tic tac toe

            args: lr          - learning rate
                  rand_choice - initial probability of choosing a random action
                  gamma       - maximum discounted future reward
        """
        self.lr = lr
        self.rand_choice = rand_choice
        self.gamma = gamma

    def learn(self, env, num_episodes):
        """ learn the game using the provided enviromnent for the given number of episodes

            args: env - tic tac toe environment
                  num_episodes - number of iterations to learn from

            returns: reward - the average cumulative reward
        """

        tf.reset_default_graph()

        # build the graph
        state_input = tf.placeholder(shape=[1, env.state_space], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([env.state_space, env.action_space], 0, 0.01))
        q_out = tf.matmul(state_input, W)
        predict = tf.argmax(q_out, 1)
        
        target_q = tf.placeholder(shape=[1, env.action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(target_q - q_out))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        updateModel = trainer.minimize(loss)
        
        r_list = []
        # reuse single array
        state_feed = state_array(env.state_space, 0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_episodes):
                # Reset environment and get first new observation
                s = env.reset()
                # cumulateive reward
                cumulative_reward = 0
                done = False
                j = 0
                #The Q-Network
                while j < 5:
                    j += 1
                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    a, allQ = sess.run([predict, q_out], feed_dict={state_input: state_feed})
                    # may choose invalid plays, if so pick random valid one
                    if not env.can_play(a[0]) or np.random.rand(1) < self.rand_choice:
                        a[0] = random_choice(env)
                    # Get new state and reward from environment
                    s1, reward, done = env.play_X(a[0])
                    #Obtain the Q' values by feeding the new state through our network
                    Q1 = sess.run(q_out, feed_dict={state_input: state_feed})
                    #Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0,a[0]] = reward + self.gamma * maxQ1
                    #Train our network using target and predicted Q values
                    _,W1 = sess.run([updateModel, W], feed_dict={state_input: state_feed, target_q:targetQ})
                    cumulative_reward += reward
                    # deactivate last state and activate next
                    state_feed[0][s] = 0.0
                    state_feed[0][s1] = 1.0
                    s = s1
                    
                    if done:
                        #Reduce chance of random action as we train the model.
                        self.rand_choice = 1./((i / 50) + 10)
                        break
                    # now play random play for O
                    _, _, done = env.play_O(random_choice(env))
                    if done:
                        break
                # end while
                r_list.append(cumulative_reward)
            # end for
        # average cumulative reward
        return sum(r_list)/num_episodes
