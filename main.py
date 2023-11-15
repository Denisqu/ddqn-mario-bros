import utils.env_tuner as env_tuner
import utils.logging as logging
import core.agent as agent
from pathlib import Path
import torch
import random, datetime, os, copy


if __name__ == '__main__':
    # получаем среду со всеми преобразованиями
    # (rgb -> grayscale), (отбрасывание лишних кадров), (масштабирование картинки)
    env = env_tuner.get_tuned_env()
    
    # используем вычисления на GPU, если есть возможность
    use_cuda = torch.cuda.is_available()
    checkpoint_file_rel_path = "C:/git repos/python-projects/ddqn-mario-bros/checkpoints/2023-10-14T21-18-14/mario_net_3.chkpt"
    print(f"Using CUDA: {use_cuda}")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    mario = agent.Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    if os.path.exists(f'{checkpoint_file_rel_path}'):
         checkpoint = torch.load(checkpoint_file_rel_path)
         mario.net.load_state_dict(checkpoint['model'])
         #mario.exploration_rate = checkpoint['exploration_rate'] - 0.39
         #mario.curr_step = checkpoint["curr_step"]
    else:
        print(f"no checkpoint found - ({checkpoint_file_rel_path})")
        
    logger = logging.MetricLogger(save_dir)

    episodes = 40000
    for e in range(episodes):
        state = env.reset()
        # Играем в игру!
        while True:

            # Запрашиваем прогноз наиболее оптимального действия для
            # среды в текущей момент
            action = mario.act(state)
            
            # Agent performs action

            # Агент совершает действие и получает
            # от среды следующее состояние среды и текущую награду
            next_state, reward, done, info = env.step(action)
            env.render()

            # Агент кэширует полученный опыт в Replay Buffer
            mario.cache(state, next_state, action, reward, done)

            # Агент подсчитывает среднее значение q-функции и функцию потерь
            # для случайной выборки опыта из Replay Buffer 
            q, loss = mario.learn()

            # Логируем наши значения обучения
            logger.log_step(reward, loss, q)
            
            # Update state

            # Обновляем текущее значение переменной среды
            state = next_state

            # Проверяем закончилась ли игра
            if done or info["flag_get"]:
                break
        
        
        logger.log_episode()
        if e % 2 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        if e % 10 == 0:
            mario.save()
            
        

    

    

