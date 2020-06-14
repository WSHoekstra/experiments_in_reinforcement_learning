# -*- coding: utf-8 -*-
# import os
# os.chdir('C:/Users/Walter/Documents/GitHub/experiments_in_reinforcement_learning')
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from experiments.agent import Agent
from experiments.memorybank import MemoryBank

    
class DQNAgent(Agent):
    def make_model(self, activation_function='relu'):
        model = Sequential()
        model.add(Dense(units=self.observation_space_size, input_dim=self.observation_space_size, name='input')) # state + actions
        model.add(Dense(units= (2 * self.observation_space_size), activation=activation_function)) # hidden
        model.add(Dense(units=self.observation_space_size, activation=activation_function)) # hidden
        model.add(Dense(units=self.action_space_size, activation=activation_function, name='output')) # output the actions
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        return model
    
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space_size)
        predicted_action_values = self.model.predict(state)
        return np.argmax(predicted_action_values[0])
    
    
    def learn_from_memories(self):
        if len(self.memorybank.experiences) < self.batch_size:
            return  # dont start learning before we have enough observations
        minibatch = self.memorybank.retrieve_random_experiences(self.batch_size)
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
    

    def __init__(self, 
                 observation_space_size, 
                 action_space_size, 
                 model_filepath,
                 epsilon=0.999,
                 epsilon_decay=0.95,
                 gamma=0.95,
                 epsilon_min=0.01,
                 memory_size=20000, 
                 training_data_size=512,
                 batch_size=32,
                 learning_rate=0.01,
                 load_model_from_disk=True):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model_filepath = model_filepath
        if load_model_from_disk:
            try:
                self._model = load_model(self.model_filepath)
            except:
                pass