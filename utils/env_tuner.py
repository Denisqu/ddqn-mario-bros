from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def get_tuned_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = max_and_skip_env(env)
    env = process_frame84(env)
    env = image_to_pytorch(env)
    env = buffer_wrapper(env, 4)
    env = scaled_float_frame(env)
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


def max_and_skip_env(env):
    """Every action the agent makes is repeated over 4 frames"""
    return env

def process_frame84(env):
    """The size of each frame is reduced to 84×84"""
    return env

def image_to_pytorch(env):
    """The frames are converted to PyTorch tensors"""
    return env

def buffer_wrapper(env, x):
    """Only every x frame is collected by the buffer"""
    return env

def scaled_float_frame(env):
    """The frames are normalized so that pixel values are between 0 and 1"""
    return env

