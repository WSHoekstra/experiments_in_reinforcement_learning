# -*- coding: utf-8 -*-
import os
# os.chdir('C:/Users/Walter/Documents/GitHub/experiments_in_reinforcement_learning')
import random
import collections
import numpy as np


class MemoryBank:
    '''
    A buffer to store the experiences our agent gets from the environment.
    '''
    def __init__(self, 
                 experiences_memory_size=2000, 
                 rewards_memory_size=None):
        self.experiences_memory_size = experiences_memory_size
        self.experiences = collections.deque(maxlen=experiences_memory_size)
        self.rewards_memory_size = rewards_memory_size
        self.rewards = collections.deque(rewards_memory_size) if rewards_memory_size else collections.deque() # default to unlimited
            
    def commit_experience_to_memory(self, state, action, reward, next_state, done):
        '''
        Commit a memory to the memory
        '''
        self.experiences.append((state, action, reward, next_state, done))    
    
    def commit_rewards_to_memory(self, reward): 
        '''
        Commit a memory to the memory
        '''
        self.rewards.append(reward)
    
    def retrieve_random_experiences(self, n):            
        n = min(n, len(self.experiences)) # cant retrieve more than stored        
        return random.sample(self.experiences, n)   
    
    def calculate_running_avg_of_recent_rewards(self, n):
        rewards = list(self.rewards)[-n:]
        return np.asarray(rewards).mean()

# memorybank = MemoryBank()
