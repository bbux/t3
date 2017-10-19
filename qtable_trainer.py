# -*- coding: utf-8 -*-
""" module holding the qtable trainer class """
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
    q_available = [-10] * 9
    for i in available_actions:
        q_available[i] = Q[s, i]
    noise = np.random.randn(1, len(q_available)) * (1./(i + 1))
    best_choice = np.argmax(q_available + noise)

    return best_choice


def pick_best_move(Q, s, env):
    """ pick an best move from available ones

        args: Q   - the q learning matrix
              s   - the current state
              env - the tic tac toe environment

        returns: the best move
    """
    # what board moves can we chose from
    available_actions = env.available_actions()
    # current rewards forthe available actions
    q_available = [-10] * 9
    for i in available_actions:
        q_available[i] = Q[s, i]
    best_choice = np.argmax(q_available)

    return best_choice


def random_choice(env):
    """ makes a random move choice from available ones

        args: env - the tic tac toe environment

        returns: a random move choice
    """
    return np.random.choice(env.available_actions())


class QTableTrainer(object):
    """ class for encapsulating learning the game of tic tac toe """
    def __init__(self, env, lr=0.8, gamma=0.95):
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

        # prime learning with trying all moves as first move
        for s in env.available_actions():
            env.reset()
            self.run_episode(env, s, 1)

        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            # cumulateive reward
            cumulative_reward = self.run_episode(env, s, i)
            # end while
            rList.append(cumulative_reward)
        # end for
        return sum(rList)/num_episodes

    def run_episode(self, env, first_move, i):
        # cumulateive reward
        cumulative_reward = 0
        done = False
        j = 0
        s = first_move
        #The Q-Table learning algorithm
        while j <= 5:
            j += 1
            # Choose an action by greedily (with noise) picking from Q table
            a = pick_best_available_action(self.Q, s, env, i)
            # Get new state and reward from environment
            s1, x_reward, done = env.play_X(a)
            # Update Q-Table with new knowledge
            self.Q[s, a] = self.Q[s, a] + self.lr*(x_reward + self.gamma*np.max(self.Q[s1, :]) - self.Q[s, a])
            cumulative_reward += x_reward
            if done:
                break
            # now play random play for O
            # Update Q-Table with new knowledge from o play
            s2, o_reward, done = env.play_O(random_choice(env))
            self.Q[s, a] = self.Q[s, a] + self.lr*((x_reward + o_reward) + self.gamma*np.max(self.Q[s2, :]) - self.Q[s, a])
            cumulative_reward += o_reward

            s = s2

            if done:
                break

        return cumulative_reward

    def play(self, env):
        """ uses the learned policy to play tic tac to with the given environment """
        s = 0
        while True:
            move = pick_best_move(self.Q, s, env)
            s, reward, done = env.play_X(int(move))
            if reward != 0:
                print("X won!")

            if done:
                break;

            env.print_state()
            available = env.available_actions()
            move = input("Enter Move for O. Available: " + str(available) + "\n")
            if int(move) not in available:
                raise Exception("Invalid Move")

            s, reward, done = env.play_O(int(move))
            if reward != 0:
                print("O won!")

            if done:
                print("Its a tie")
                break;

    def reset(self):
        """ reset the values for the Q Table """
        self.Q.fill(0)
