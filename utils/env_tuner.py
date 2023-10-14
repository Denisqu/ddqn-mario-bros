from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

from utils.wrappers import obs_wrappers as o_w
from utils.wrappers import custom_wrappers as c_w
from gym.wrappers import FrameStack

def get_tuned_env():
    #if gym.__version__ < '0.26':
    #    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    #else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = c_w.SkipFrame(env, skip=4)
    env = o_w.GrayScaleObservation(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    """FrameStack is a wrapper that allows us to squash
    consecutive frames of the environment into a
    single observation point to feed to our learning model.
    This way, we can identify if Mario was landing or jumping
    based on the direction of his movement in the previous
    several frames."""
    env = FrameStack(env, num_stack=4)
    return env
