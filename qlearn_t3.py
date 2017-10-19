# -*- coding: utf-8 -*-
""" module for running qlearn tests on tic tac toe environments """
import t3
import time
from qtable_trainer import QTableTrainer
from qnetwork_trainer import QNetworkTrainer

def train(trainer, env):
    start = time.time()
    average_cumulative_reward = trainer.learn(env, 10000)
    print("Score over time: %f" % average_cumulative_reward)
    print("Training Time: %f" % (time.time() - start))
    env.reset()

#    trainer.play(env)

    wins = 0
    losses = 0
    ties = 0
    for i in range(1000):
        env.reset()
        reward = trainer.play_self(env)
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            ties += 1
            
    print("Wins: %d, Losses: %d, Ties %d" % (wins, losses, ties))

def main():
    # the tic tac toe environment
    t3env = t3.make()
    print("Training with QTable...")
    train(QTableTrainer(t3env), t3env)
    print("Training with QNetwork...")
    #train(QNetworkTrainer(), t3env)


if __name__ == "__main__":
    main()