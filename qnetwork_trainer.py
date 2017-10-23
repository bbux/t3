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
    def __init__(self, lr=0.01, rand_choice=0.1, gamma=0.99):
        """ Creates a trainer that uses a tensorflow neural net to learn how to
            play tic tac toe

            args: lr          - learning rate
                  rand_choice - initial probability of choosing a random action
                  gamma       - maximum discounted future reward
        """
        self.lr = lr
        self.rand_choice = rand_choice
        self.gamma = gamma

    def build(self, env):
        tf.reset_default_graph()

        # build the graph
        self.state_input = tf.placeholder(shape=[1, env.state_space], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([env.state_space, env.action_space], 0, 0.01))
        self.q_out = tf.matmul(self.state_input, self.W)
        self.predict = tf.argmax(self.q_out, 1)
        
        self.target_q = tf.placeholder(shape=[1, env.action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.target_q - self.q_out))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updateModel = trainer.minimize(loss)
        # reuse single array
        self.state_feed = state_array(env.state_space, 0)
        self.saver = tf.train.Saver({'W': self.W})

        
    def learn(self, env, num_episodes):
        """ learn the game using the provided enviromnent for the given number of episodes

            args: env - tic tac toe environment
                  num_episodes - number of iterations to learn from

            returns: reward - the average cumulative reward
        """

        # build the graph
        self.build(env)
        
        r_list = []
        state_action = {}
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_episodes):
                # reset input vector
                self.state_feed.fill(0)
                # Reset environment and get first new observation
                s = env.reset()
                # cumulateive reward
                cumulative_reward = 0
                done = False
                j = 0
                #The Q-Network
                while j <= 5:
                    j += 1
                    # set input vector to activate current state
                    self.state_feed[0][s] = 1.0

                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    a, allQ = sess.run([self.predict, self.q_out], feed_dict={self.state_input: self.state_feed})
                    # store the state to action mapping for later
                    state_action[s] = a
                    
                    # may choose invalid plays, if so pick random valid one
                    if not env.can_play(a[0]) or np.random.rand(1) < self.rand_choice:
                        a[0] = random_choice(env)
                    # Get new state and reward from environment
                    s1, x_reward, done = env.play_X(a[0])

                    #Obtain the Q' values by feeding the new state through our network
                    Q1 = sess.run(self.q_out, feed_dict={self.state_input: self.state_feed})

                    #Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)
                    targetQ = allQ

                    o_reward = 0
                    if not done:
                        # now play random play for O
                        s2, o_reward, done = env.play_O(random_choice(env))
                    
                    targetQ[0,a[0]] = x_reward + o_reward + self.gamma * maxQ1

                    #Train our network using target and predicted Q values
                    _,W1 = sess.run([self.updateModel, self.W], feed_dict={self.state_input: self.state_feed, self.target_q: targetQ})
                    cumulative_reward += x_reward + o_reward

                    # deactivate last state and activate next
                    self.state_feed[0][s] = 0.0
                    s = s2
                    
                    if done:
                        #Reduce chance of random action as we train the model.
                        self.rand_choice = 1./((i / 50) + 10)
                        break
                    
                # end while
                r_list.append(cumulative_reward)
            # end for
            self.saver.save(sess, '/tmp/qnetwork_weights.tf')
        # reset input vector
        self.state_feed.fill(0)
        # average cumulative reward
        return sum(r_list)/num_episodes

    def pick_best_move(self, s, env):
        """ pick an best move from available ones
    
            args: s   - the current state
                  env - the tic tac toe environment
    
            returns: the best available move
        """
        # what board moves can we chose from
        available_actions = env.available_actions()
        self.state_feed[0][s] = 1.0
        # current rewards forthe available actions
        q_available = [-10] * 9
        for i in available_actions:
            q_available[i] = 0

        with tf.Session() as session:
            #sess.run(tf.global_variables_initializer())
            self.saver.restore(session, '/tmp/qnetwork_weights.tf')
            a, allQ = session.run([self.predict, self.q_out], feed_dict={self.state_input: self.state_feed})
            best_choice = np.argmax(q_available + allQ)
    
        # unset input action
        self.state_feed[0][s] = 0.0
        
        return best_choice