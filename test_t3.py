import t3
import pytest


def test_reset():
    env = t3.make()
    env.play_X(1)
    state_index = env.reset()
    assert state_index == 0


def test_error_in_range1():
    with pytest.raises(Exception):
        env = t3.make()
        env.play_X(10)


def test_error_in_range2():
    with pytest.raises(Exception):
        env = t3.make()
        env.play_X(-1)


def test_error_already_play_ed():
    with pytest.raises(Exception):
        env = t3.make()
        env.play_X(1)
        env.play_O(1)


def eval_win(x_moves, o_moves):
    env = t3.make()
    for i in range(len(x_moves) - 1):
        _, reward, done = env.play_X(x_moves[i])
        assert reward == 0
        assert done is False
        _, reward, done = env.play_O(o_moves[i])
        assert reward == 0
        assert done is False
    
    # winning move
    _, reward, done = env.play_X(x_moves[-1])
    assert reward == env.win_reward
    assert done is True
    

def eval_lose(x_moves, o_moves):
    env = t3.make()
    for i in range(len(x_moves) - 1):
        _, reward, done = env.play_X(x_moves[i])
        assert reward == 0
        assert done is False
        _, reward, done = env.play_O(o_moves[i])
        assert reward == 0
        assert done is False
    
    # last x move
    _, reward, done = env.play_X(x_moves[-1])
    assert reward == 0
    assert done is False
    # o's winning move
    _, reward, done = env.play_O(o_moves[-1])
    assert reward == env.lose_reward
    assert done is True
    

def test_win_1():
    """
     X X X
     O O _
     _ _ _
    """
    eval_win(x_moves=[0, 1, 2],
             o_moves=[3, 4])


def test_win_2():
    """
     X O X
     O X O
     O X X
    """
    eval_win(x_moves=[0, 2, 4, 7, 8],
             o_moves=[1, 3, 5, 6])


def test_lose_1():
    """
     X X _
     O O O
     X _ _
    """
    eval_lose(x_moves=[0, 1, 6],
              o_moves=[3, 4, 5])

def test_lose_2():
    """
     X X O
     _ O _
     O X _
    """
    eval_lose(x_moves=[0, 1, 7],
              o_moves=[2, 4, 6])
