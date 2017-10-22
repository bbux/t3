# -*- coding: utf-8 -*-
""" module for running qlearn tests on tic tac toe environments """
import t3
import time
from qtable_trainer import QTableTrainer
from qnetwork_trainer import QNetworkTrainer
from player import Player

def train(trainer, env):
    start = time.time()
    average_cumulative_reward = trainer.learn(env, 5000)
    print("Score over time: %f" % average_cumulative_reward)
    print("Training Time: %f" % (time.time() - start))
    env.reset()

#    trainer.play(env)



def main():
    # the tic tac toe environment
    t3env = t3.make()
    print("Training with QTable...")
    #table_trainer = QTableTrainer(t3env)
    #train(table_trainer, t3env)
    
    print("Training with QNetwork...")
    network_trainer = QNetworkTrainer()
    train(network_trainer, t3env)

    player = Player(network_trainer)
    #player.evaluate(t3env, 1000)
    

if __name__ == "__main__":
    main()