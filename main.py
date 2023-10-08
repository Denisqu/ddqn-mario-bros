import utils.env_tuner as env_tuner

if __name__ == '__main__':
    env = env_tuner.get_tuned_env()
    
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()
