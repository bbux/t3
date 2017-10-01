# -*- coding: utf-8 -*-
"""
Modlue for creating and interacting with tic tac toe (t3) board

Modeled off of openai gym: https://gym.openai.com/docs/

@author: Brian Buxton
"""

X_MARKER = 1
O_MARKER = 2
EMPTY_MARKER = 0
WIN_POSITIONS = [
    [0, 1, 2], [3 ,4, 5], [6, 7, 8],
    [0, 3 ,6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2 ,4, 6]
]

class Env(object):
    """ Class for holding tic tac toe environment """

    def __init__(self, win_reward=1):
        """ conscructs a tic tac toe environment

            args - win_reward - reward for winning, default is 1
        """
        self.reset()
        self.reward = win_reward
        self.state_space = 3**9
        self.action_space = 9

    def reset(self):
        """ resets board state
            returns: observation - the reset state of the board
        """
        # creates and empty board
        self.state = [EMPTY_MARKER] * 9
        self.game_winner = None
        return self._calculate_state_index()

    def available_actions(self):
        """ returns the set of actions/board positions that do not have a
            marker (X or O) on them
        """
        return [i for i, v in enumerate(self.state) if v == EMPTY_MARKER]

    def play_X(self, position):
        """ play X at given position

        args: position to play X marker

        return: state_index - after playing
                reward      - if any (default is 1 for win, zero otherwise)
                done        - if game is over
        """
        return self._play(position, X_MARKER)

    def play_O(self, position):
        """ play O at given position

        args: position to play O marker

        return: state_index - after playing
                reward      - if any (default is 1 for win, zero otherwise)
                done        - if game is over
        """
        return self._play(position, O_MARKER)

    def _play(self, position, marker):
        """ play the marker at given positon """
        if position not in range(0, 9):
            raise Exception("Position: %d is not valid! Must be 0-8." % position)

        if self.state[position] != EMPTY_MARKER:
            raise Exception("Position %d is not empty!" % position)

        self.state[position] = marker
        reward = self._calculate_reward(marker)

        return (self._calculate_state_index(), reward, self._game_is_done())

    def _calculate_reward(self, marker):
        """ if this is a win state for the given marker

            args: marker - X, or, O to check for win
        """
        all_same = [marker] * 3
        # check each winning position, assume no win
        for wins in WIN_POSITIONS:
            vals = [self.state[i] for i in wins]
            # i.e [1, 1, 1] or [2, 2, 2]
            if vals == all_same:
                return self.reward

        return 0

    def _game_is_done(self):
        """ is the game over, x wins, o wins, or draw """
        x_reward = self._calculate_reward(X_MARKER)
        if x_reward != 0:
            return True

        o_reward = self._calculate_reward(O_MARKER)
        if o_reward != 0:
            return True

        # last check, no more moves left
        no_moves_left = EMPTY_MARKER not in self.state
        return no_moves_left

    def _calculate_state_index(self):
        """ given the state of the board, where the x's and o's are, what index
        is this in the state space.  There are 3^9 possible states although not
        all states are realizable in a real game, i.e. X fills all squares.
        """
        # each square in the board is stored as a base 3 value, 0, 1, 2, these
        # can then be used to create a base 3 number which represents the current
        # state index of the board
        state_as_string = "".join(str(e) for e in self.state)
        # python has build in base conversion
        # here we are converting  a number of the form 210012102 to the index between
        # 0 and 3^9 that it corresponds to
        return int(state_as_string, 3)

def make():
    """ make a tic tac toe environment """
    return Env()
