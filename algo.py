import torch,sys,gymnasium_sudoku,random,time
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class MRV: 
    def __init__(self,state): 
        self.state = torch.as_tensor(state)

    def get_region(self,idx):
        row,col = idx

        x_list = self.state[row].tolist()   
        x_list.pop(col)

        y_list = self.state[:,col].tolist() 
        y_list.pop(row)
        
        block_idx = (row // 3) * 3 + (col // 3)
        block = self.state.reshape(3,3,3,3).permute(0,2,1,3).reshape(9,9)[block_idx].tolist()
        block_row = row % 3 ; block_col = col % 3
        cell_idx = block_row * 3 + block_col
        block.pop(cell_idx)
        
        region = torch.tensor([x_list + y_list + block]).unique()
        region = region[region!=0]
        return region

    def update_domain(self,idx):
        self.idx = torch.as_tensor(idx)
        self.domain = torch.arange(1,10).repeat(self.idx.size(0),1)
        self.dic = torch.cat([self.idx,self.domain],-1) # -> column[1-2] = indice , column[3-11] = domain

        for tensor in self.dic:
            idx = tensor[:2]
            domain = tensor[2:]

            region = self.get_region(idx)
            filler = torch.full((domain.size(0) - region.size(0),),0)
            region = torch.cat([region,filler])
        
            domain_mask = torch.isin(domain,region)
            domain = torch.masked_fill(domain,domain_mask,-1)

            tensor[2:] = domain

    def get_minimum_value(self):
        value_tensor = torch.empty(self.dic.size(0)).long()
        for i,tensor in enumerate(self.dic):
            domain = tensor[2:]
            value = (domain > 0).sum()
            value_tensor[i] = value  
        return value_tensor.squeeze()

    def sample_action(self,idx): 
        self.update_domain(idx) 
        vals = self.get_minimum_value()
        min_vals = vals.min()
        x = (vals == min_vals).nonzero()
        sample_idx = random.choices(x)
        cell = self.dic[:,:2][tuple(sample_idx)]
        
        cell_value = self.dic[:,2:][tuple(sample_idx)].squeeze()
        cell_value = cell_value[cell_value > 0]
    
        if len(cell_value) > 1:
            cell_value = torch.as_tensor(random.choices(cell_value.tolist()))

        action = torch.cat([cell.squeeze(),cell_value]).numpy()
        
        if self.dic.size(0) == 1: # handling last cell remaining
            dic = self.dic.squeeze()
            cell = dic[:2]
            domain = dic[2:]
            domain = domain[domain > 0]
            action = torch.cat([cell,domain]).numpy()

        return action

def env(horizon=None):
    x = gym.make("sudoku-v1",mode="easy",horizon=100,render_mode="human")
    return x

if __name__ == "__main__":
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    env = env()
    state = torch.as_tensor(env.reset()[0])
    
    for n in tqdm(range(int(1e3))):
        idx = (torch.as_tensor(state) == 0).nonzero()
        heuristic = MRV(state)
        
        state,reward,done,trunc,info = env.step(heuristic.sample_action(idx))
        env.render()
        time.sleep(0.1)
        
        if done:
            _done = torch.as_tensor(state) == torch.as_tensor(env.unwrapped.solution)
            print("Done == ",torch.all(_done).item())
            state = torch.as_tensor(env.reset()[0])
        
        
