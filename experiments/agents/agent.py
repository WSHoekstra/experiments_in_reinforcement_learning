# -*- coding: utf-8 -*-
from experiments.memorybank import MemoryBank


class Agent:    
    def __init__(self, memory_size=None, model_filepath=None):
        self._model = None
        self.model_filepath = model_filepath
        self.memorybank = MemoryBank(memory_size) if memory_size else MemoryBank()
    
    def make_model(self):
        '''
        Placeholder method
        '''
        return None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self.make_model()
        return self._model   

    def save_model_to_disk(self):
        self.model.save(self.model_filepath)        
        