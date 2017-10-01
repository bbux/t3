# -*- coding: utf-8 -*-
""" module for running qlearn tests on tic tac toe environments """
import t3
from qtable_trainer import QTableTrainer


def main():
    # the tic tac toe environment
    t3env = t3.make()
    trainer = QTableTrainer(t3env)
    average_cumulative_reward = trainer.learn(t3env, 2000)
    print("Score over time: " +  str(average_cumulative_reward))


if __name__ == "__main__":
    main()