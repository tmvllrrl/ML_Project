import numpy as np
import torch
from dataclasses import dataclass
import random
import os
import math
from config import *

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None

class ReplayMemory:
    def __init__(self, args):
        self._args = args
        self._memory = []
        # Pointer to end of memory
        self._cur_pos = 0
    
    def append(self, e_t):
        """Append experience."""
        if len(self._memory) >= self._args.capacity:
            self._memory[self._cur_pos] = e_t
        else:
            self._memory.append(e_t)
        
        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self._args.capacity 

    def sample(self):
        """Sample batch size experience replay."""
        return np.random.choice(self._memory, size=self._args.batch_size, replace=False)

    def current_capacity(self):
        return len(self._memory)



class PrioritizedExpReplay:
    """Prioritized experience replay (PER) memory."""
    def __init__(self, args):
        self.args = args
        self._sum_tree = SumTree(self.args)
        self._memory_file = os.path.join(self.args.save_dir, "memory.pt")
        self._best_memory_file = os.path.join(self.args.save_best_dir, "memory.pt")
        self._num_steps = 0

        if self.args.load:
            self.load()
    
    def add(self, exp: Experience, error: float):
        """Append experience."""
        
        priority = self._compute_priority(error)
        self._sum_tree.add(exp, priority)

    def sample(self, batch_size: int):
        """Sample batch size experience replay."""
        segment = self._sum_tree.total() / batch_size
        priorities = torch.zeros(batch_size).to(device)
        exps = []
        indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a, b)
            p, e_i, tree_idx = self._sum_tree.get(mass)
            priorities[i] = p
            exps.append(e_i)
            indices.append(tree_idx)

        # Compute importance sampling weights
        sample_ps = priorities / self._sum_tree.total()
        
        # Increase per beta by the number of episodes that have elapsed
        cur_per_beta = self.args.per_beta#min(self.args.per_beta + (num_updates / self.args.decay_episodes) * (1 - self.args.per_beta) , 1.0)

        is_ws = (sample_ps  * self.cur_cap()) ** -cur_per_beta
        

        # Normalize to scale the updates downwards
        is_ws  = is_ws / is_ws.max()

        return is_ws, exps, indices

    def cur_cap(self):
        return self._sum_tree.cur_cap()

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = self._compute_priority(error)
            self._sum_tree.update(idx, priority)

    def save(self):
        model_dict = {
            "memory" : self._sum_tree.memory,
            "tree" : self._sum_tree.tree,
            "pos" : self._sum_tree._end_pos
        }
        torch.save(model_dict, self._memory_file)
    
    def save_best(self):
        model_dict = {
            "memory" : self._sum_tree.memory,
            "tree" : self._sum_tree.tree,
            "pos" : self._sum_tree._end_pos
        }
        torch.save(model_dict, self._best_memory_file)
        
    def load(self):
        if os.path.exists(self._memory_file):
            model_dict = torch.load(self._memory_file)
            self._sum_tree.memory = model_dict["memory"]
            self._sum_tree.tree = model_dict["tree"]            
            self._sum_tree._end_pos = model_dict["pos"]

    def _compute_priority(self, td_error):
        per_alpha = min((1-self.args.per_alpha) * (self._num_steps/self.args.epsilon_decay) + self.args.per_alpha, 1.0)
        return (abs(td_error) + self.args.eps) ** per_alpha 



class SumTree:
    """Sum Tree used for PER."""
    def __init__(self, args):
        self.args = args
        # Raise to next power of 2 to make full binary tree
        self.capacity = 2 ** math.ceil(
            math.log(self.args.mem_cap,2))

        # sum tree 
        self.tree = torch.zeros(2 * self.capacity - 1).to(device)
        self.memory = []

        # Pointer to end of memory
        self._end_pos = 0
    
    def add(self, exp, priority):
        """Add experience to sum tree."""
        

        cur_mem = self.memory
        end_pos = self._end_pos
        cur_cap = self.capacity
        self._end_pos = (self._end_pos + 1) % cur_cap
        idx = self.capacity + end_pos  - 1
        
        # Add experience to memory
        if len(cur_mem) < cur_cap:
            cur_mem.append(exp)
        else:
            cur_mem[end_pos] = exp
    
        # Update sum tree
        self.update(idx, priority)



    def update(self, idx, priority):
        """Update priority of element and propagate through tree."""
        # Compute priority difference
        diff = priority - self.tree[idx]

        # Propagate update through tree
        while idx >= 0:
            self.tree[idx] += diff
            # Update to parent idx
            idx = (idx - 1) // 2

    def total(self):
        return self.tree[0]

    def get(self, val):
        """Sample from sum tree based on the sampled value."""
        tree_idx = self._retrieve(val)
        data_idx = tree_idx - self.capacity + 1

        #data = self.memory[data_idx]
        try:

            data = self.memory[data_idx]
        except Exception:
            print("self.tree[tree_idx]", self.tree[tree_idx], "len(self.tree)", len(self.tree))
            print("data_idx", data_idx, "tree_idx", tree_idx, self.capacity, val, len(self.memory))
            print("self.tree", self.tree)
            import sys
            sys.exit()
        return self.tree[tree_idx], data, tree_idx

    def _retrieve(self, val):
        idx = 0
        # The left and right children
        left = 2 * idx + 1
        right = 2 * idx + 2

        # Keep going down the tree until leaf node with correct priority reached
        while left < len(self.tree):
            if val <= self.tree[left] or self.tree[left].isclose(val, atol=1e-2) or not self.tree[right].is_nonzero():
                idx = left
            else:
                idx = right
                val -= self.tree[left]

            left = 2 * idx + 1
            right= 2 * idx + 2

        return idx

    def cur_cap(self):
        return len(self.memory)
                                    
