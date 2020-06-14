# -*- coding: utf-8 -*-
# import os
# os.chdir('C:/Users/Walter/Documents/GitHub/experiments_in_reinforcement_learning')
import gym
import numpy as np
from experiments.agents.dqn import DQNAgent


env = gym.make('CartPole-v0')
state = env.reset()

agent = DQNAgent(observation_space_size=env.observation_space.shape[0],
              action_space_size=env.action_space.n,
              load_model_from_disk=True,
              model_filepath='experiments/01_cartpole/model.h5')

done = False
max_n_steps = 2000
render_every_n_episodes = 20
save_model_every_n_episodes = 10

episodes = range(1000)
for episode in episodes:
    episode_rewards = 0
    state = np.reshape(env.reset(), [1, agent.observation_space_size])
    done = False
    step_i = 0
    while step_i < max_n_steps and not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1,agent.observation_space_size])
        agent.memorybank.commit_experience_to_memory(state, action, reward, next_state, done)        
        episode_rewards += reward        
        if episode % render_every_n_episodes == 0:
            env.render()        
        step_i += 1        
        state = next_state
        if done:
            agent.memorybank.commit_rewards_to_memory(episode_rewards)
            calculate_avg_rewards_over_n_episodes = 100
            running_avg_rewards = agent.memorybank.calculate_running_avg_of_recent_rewards(calculate_avg_rewards_over_n_episodes)
            print(f'episode {episode} / epsilon {agent.epsilon} / reward: {episode_rewards} / running avg rewards {running_avg_rewards} ({calculate_avg_rewards_over_n_episodes} episodes)')
    agent.learn_from_memories()
    if episode % save_model_every_n_episodes == 0 and episode > 0:
            agent.save_model_to_disk()
env.close()

