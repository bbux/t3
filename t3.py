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

    def __init__(self, win_reward=1, lose_reward=-1, player=X_MARKER):
        """ conscructs a tic tac toe environment

            args - win_reward   - reward for winning, default is 1
                   loose_reward - reward for losing, default is -1
                   player       - who is trying to win, 1 for X, 2 for O
        """
        self.reset()
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.player = player
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
                win_reward  - if any (default is 1 for win, -1 lose, 0 otherwise)
                done        - if game is over
        """
        return self._play(position, X_MARKER)

    def play_O(self, position):
        """ play O at given position

        args: position to play O marker

        return: state_index - after playing
                win_reward  - if any (default is 1 for win, -1 for lose, zero otherwise)
                done        - if game is over
        """
        return self._play(position, O_MARKER)

    def can_play(self, position):
        """ is the given position playable, i.e. no X or O on it """
        return self.state[position] == EMPTY_MARKER

    def print_state(self):
        """ prints out the board in ascii art """
        row1 = [self._get_mark(m) for m in self.state[0:3]]
        row2 = [self._get_mark(m) for m in self.state[3:6]]
        row3 = [self._get_mark(m) for m in self.state[6:9]]
        print(" ".join(row1) + "\n")
        print(" ".join(row2) + "\n")
        print(" ".join(row3) + "\n")

    def _get_mark(self, marker):
        if marker == X_MARKER:
            return "X"
        elif marker == O_MARKER:
            return "O"
        else:
            return "_"

    def _play(self, position, marker):
        """ play the marker at given positon """
        if position not in range(0, 9):
            raise Exception("Position: %d is not valid! Must be 0-8." % position)

        if not self.can_play(position):
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
                if marker == self.player:
                    return self.win_reward
                else:
                    return self.lose_reward
        
        return 0

    def _game_is_done(self):
        """ is the game over, x wins, o wins, or draw """
        x_win_reward = self._calculate_reward(X_MARKER)
        if x_win_reward != 0:
            return True

        o_win_reward = self._calculate_reward(O_MARKER)
        if o_win_reward != 0:
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
