import t3
import pytest

def test_reset():
    env = t3.make()
    env.playX(1)
    env.reset()
    assert [0] * 9 == env.state

def test_error_in_range1():
    with pytest.raises(Exception):
        env = t3.make()
        env.playX(10)
        
def test_error_in_range2():
    with pytest.raises(Exception):
        env = t3.make()
        env.playX(-1)

def test_error_already_played():
    with pytest.raises(Exception):
        env = t3.make()
        env.playX(1)
        env.playO(1)

def test_win_1():
    env = t3.make()
    state, reward, done = env.playX(0)
    assert reward == 0
    assert done is False
    state, reward, done = env.playO(3)
    assert reward == 0
    assert done is False
    state, reward, done = env.playX(1)
    assert reward == 0
    assert done is False
    state, reward, done = env.playO(4)
    assert reward == 0
    assert done is False
    # winning move
    state, reward, done = env.playX(2)
    assert reward == env.reward
    assert done is True
    
def test_win_2():
    env = t3.make()
    state, reward, done = env.playX(0)
    assert reward == 0
    assert done is False
    state, reward, done = env.playO(3)
    assert reward == 0
    assert done is False
    state, reward, done = env.playX(1)
    assert reward == 0
    assert done is False
    state, reward, done = env.playO(4)
    assert reward == 0
    assert done is False
    state, reward, done = env.playX(6)
    assert reward == 0
    assert done is False
    # winning move
    state, reward, done = env.playO(5)
    assert reward == env.reward
    assert done is True