import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,s_cur,a,r,s_nxt,after_bert):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (s_cur,a,r,s_nxt,after_bert)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.__len__()<batch_size:
            return [],[],[],[],[]
        batch=random.sample(self.memory, batch_size)
        s_cur=[_[0] for _ in batch]
        a=[_[1] for _ in batch]
        r=[_[2] for _ in batch]
        s_nxt=[_[3] for _ in batch]
        ab=[_[4] for _ in batch]
        return s_cur,a,r,s_nxt,ab
    
    def __len__(self):
        return len(self.memory)