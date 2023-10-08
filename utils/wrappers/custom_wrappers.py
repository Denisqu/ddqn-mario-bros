import gym


class SkipFrame(gym.Wrapper):
    """SkipFrame is a custom wrapper that inherits from
    gym.Wrapper and implements the step() function.
    Because consecutive frames don’t vary much, we can skip
    n-intermediate frames without losing much information.
    The n-th frame aggregates rewards accumulated over
    each skipped frame."""
    
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info