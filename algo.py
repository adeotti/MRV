import torch,sys,os,gymnasium_sudoku,mlflow,random,math
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np


class MRV: 
    def __init__(self,state,idx): 
        self.state = torch.as_tensor(state)
        self.idx = idx
        self.domain = torch.arange(1,10).repeat(self.idx.size(0),1)
        self.dic = torch.cat([self.idx,self.domain],-1) # -> column[1-2] = indice , column[3-11] = domain
        
    def get_region(self,idx):
        row,col = idx

        x_list = self.state[row].tolist()   ; x_list.pop(row)
        y_list = self.state[:,col].tolist() ; y_list.pop(col)

        block_idx = (row // 3) * 3 + (col // 3)
        block = self.state.reshape(3,3,3,3).permute(0,2,1,3).reshape(9,9)[block_idx].tolist()
        block_row = row % 3 ; block_col = col % 3
        cell_idx = block_row * 3 + block_col
        block.pop(cell_idx)
        
        region = torch.tensor([x_list + y_list + block]).unique().nonzero().squeeze()
        return region

    def update_domain(self):
        for tensor in self.dic:
            idx = tensor[:2]
            domain = tensor[2:]
            region = self.get_region(idx)

            filler = torch.full((domain.size(0) - region.size(0),),0)
            region = torch.cat([region,filler])
            domain_mask = (region == domain)
            domain = torch.masked_fill(domain,domain_mask,-1)

            tensor[2:] = domain 

    def get_minimum_value(self):
        value_tensor = torch.empty(self.dic.size(0)).long()
        for i,tensor in enumerate(self.dic):
            domain = tensor[2:]
            value = (domain > 0).sum()
            value_tensor[i] = value  
        return value_tensor.squeeze()

    def sample_action(self): 
        self.update_domain() 
        vals = self.get_minimum_value()
        min_vals = vals.min()
        x = (vals == min_vals).nonzero()
        sample_idx = random.choices(x)
        cell = self.dic[:,:2][sample_idx]
        n = self.dic[:,2:][sample_idx]
        n = n[n > 0]
        return torch.cat([cell.squeeze(),n]).numpy(),sample_idx[0].item()


def env(horizon=None):
    x = gym.make("sudoku-v1",mode="easy",horizon=400,render_mode="human")
    return x

if __name__ == "__main__":
    env = env()
    state = torch.as_tensor(env.reset()[0])
    idx = (state == 0).nonzero()
    for n in range(1000):
        heuristic = MRV(state,torch.as_tensor(idx))
        action,sample_idx = heuristic.sample_action()
        state,reward,done,trunc,info = env.step(action)
        env.render()
        if done:
            sys.exit("done")
            state = env.reset()
