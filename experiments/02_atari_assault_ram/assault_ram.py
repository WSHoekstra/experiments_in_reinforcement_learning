# -*- coding: utf-8 -*-
import os
# os.chdir('C:/Users/Walter/Documents/GitHub/experiments_in_reinforcement_learning')

import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import collections


env = gym.make('Assault-ram-v0')
state = env.reset()

    
class Agent:
    def make_model(self, activation_function='tanh'):
        model = Sequential()
        model.add(Dense(units=self.observation_space_size, input_dim=self.observation_space_size, name='input')) # state + actions
        model.add(Dense(units= (2 * self.observation_space_size), activation=activation_function)) # hidden
        model.add(Dense(units=self.observation_space_size, activation=activation_function)) # hidden
        model.add(Dense(units=self.action_space_size, activation='linear', name='output')) # output the actions
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        return model
    
    def commit_to_memory(self, state, action, reward, next_state, done):
        self.memorybank.append((state, action, reward, next_state, done))    
    
    def commit_episode_rewards_to_memory(self, reward): 
        self.episode_rewards.append(reward)
    
    def retrieve_random_memories(self, n):            
        n = min(n, len(self.memorybank)) # cant retrieve more than stored        
        return random.sample(self.memorybank, n)   
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space_size)
        predicted_action_values = self.model.predict(state)
        return np.argmax(predicted_action_values[0])
    
    def learn_from_memories(self):
        if len(self.memorybank) < self.training_data_size:
            return  # dont start learning before we have enough observations
        minibatch = self.retrieve_random_memories(self.training_data_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0]) * (1 - int(done))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, batch_size=self.batch_size, verbose=0)
        if  self.epsilon * self.epsilon_decay > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        
        else:
            self.epsilon = self.epsilon_min
    
    def calculate_running_avg_of_recent_rewards(self, n):
        rewards = list(self.episode_rewards)[-n:]
        return np.asarray(rewards).mean()
        
    
    @property
    def model(self):
        if self._model is None:
            self._model = self.make_model()
        return self._model
    

    def save_model_to_disk(self):
        self.model.save(self.model_filepath)
        
    
    def __init__(self, 
                 observation_space_size, 
                 action_space_size, 
                 epsilon=0.999,
                 epsilon_decay=0.95,
                 gamma=0.95,
                 epsilon_min=0.01,
                 memory_size=20000, 
                 training_data_size=512,
                 batch_size=32,
                 learning_rate=0.01,
                 load_model_from_disk=True):
        self._model = None
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.training_data_size = training_data_size
        self.memorybank = collections.deque(maxlen=memory_size)
        self.episode_rewards = collections.deque()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model_filepath = 'experiments/01_cartpole/model.h5'
        if load_model_from_disk:
            try:
                self._model = load_model(self.model_filepath)
            except:
                pass
            

agent = Agent(observation_space_size=env.observation_space.shape[0],
              action_space_size=env.action_space.n,
              load_model_from_disk=True)

# agent.model.summary()
# out = agent.model.predict( [ [0.25, 0.5, 0.75, 1, 1] ] ) # untrained network output on a reset env

done = False
max_n_steps = 2000
render_every_n_episodes = 20


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
        agent.commit_to_memory(state, action, reward, next_state, done)        
        episode_rewards += reward        
        if episode % render_every_n_episodes == 0:
            env.render()        
        step_i += 1        
        state = next_state        
        if done:            
            agent.commit_episode_rewards_to_memory(episode_rewards)            
            calculate_avg_rewards_over_n_episodes = 100
            running_avg_rewards = agent.calculate_running_avg_of_recent_rewards(calculate_avg_rewards_over_n_episodes)            
            print(f'episode {episode} / epsilon {agent.epsilon} / reward: {episode_rewards} / running avg rewards {running_avg_rewards} ({calculate_avg_rewards_over_n_episodes} episodes)')
    agent.learn_from_memories()
    if episode % 10 == 0 and episode > 0:        
            agent.save_model_to_disk()
env.close()

