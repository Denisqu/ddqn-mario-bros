import utils.env_tuner as env_tuner

if __name__ == '__main__':
    env = env_tuner.get_tuned_env()
    print(env.observation_space.sample)
