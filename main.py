import utils.env_tuner as env_tuner
import utils.logging as logging
import core.agent as agent
from pathlib import Path
import torch
import random, datetime, os, copy


if __name__ == '__main__':
    env = env_tuner.get_tuned_env()
    
    use_cuda = torch.cuda.is_available()
    checkpoint_file_rel_path = "checkpoints\2023-10-14T19-36-58\mario_net_0.chkpt"
    print(f"Using CUDA: {use_cuda}")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = agent.Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    if os.path.exists(f'.\{checkpoint_file_rel_path}'):
         checkpoint = torch.load(checkpoint_file_rel_path)
         mario.net.load_state_dict(checkpoint['model'])
         mario.exploration_rate = checkpoint['exploration_rate']
         mario.curr_step = checkpoint["curr_step"]
    else:
        print(f"no checkpoint found - ({checkpoint_file_rel_path})")
        
    logger = logging.MetricLogger(save_dir)

    episodes = 4000
    for e in range(episodes):
        state = env.reset()
        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)
            
            # Agent performs action
            next_state, reward, done, info = env.step(action)
            #env.render()

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)
            
            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break
        
        
        logger.log_episode()
        if e % 2 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        if e % 10 == 0:
            mario.save()
            
        

    

    

