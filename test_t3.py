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

def test_win_1():
    env = t3.make()
    _, reward, done = env.play_X(0)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_O(3)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_X(1)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_O(4)
    assert reward == 0
    assert done is False
    # winning move
    _, reward, done = env.play_X(2)
    assert reward == env.reward
    assert done is True

def test_win_2():
    env = t3.make()
    _, reward, done = env.play_X(0)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_O(3)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_X(1)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_O(4)
    assert reward == 0
    assert done is False
    _, reward, done = env.play_X(6)
    assert reward == 0
    assert done is False
    # winning move
    _, reward, done = env.play_O(5)
    assert reward == env.reward
    assert done is True
