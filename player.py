import numpy as np

def random_choice(env):
    """ makes a random move choice from available ones

        args: env - the tic tac toe environment

        returns: a random move choice
    """
    return np.random.choice(env.available_actions())


class Player(object):
    """ class for playing tic tac toe against the provided trainers learned policy """

    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, env, num_games):
        """ evaluate the optimacy of the learned policy by playing n games """
        wins = 0
        losses = 0
        ties = 0
        for i in range(num_games):
            env.reset()
            reward = self.play(env)
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                ties += 1
    
        print("Wins: %d, Losses: %d, Ties %d" % (wins, losses, ties))

    def play(self, env):
        """ uses the learned policy to play tic tac against the trainer """
        s = 0
        reward = 0
        while True:
            move = self.trainer.pick_best_move(s, env)
            s, reward, done = env.play_X(move)

            if done:
                break;

            s, reward, done = env.play_O(random_choice(env))

            if done:
                break;
        return reward

    def play_interactive(self, env):
        """ uses the learned policy to play tic tac interactively """
        s = 0
        while True:
            move = self.trainer.pick_best_move(s, env)
            s, reward, done = env.play_X(move)
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
