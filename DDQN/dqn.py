import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json, math, os
from config import *

from replay_memory import ReplayMemory
from replay_memory import PrioritizedExpReplay

class DQNActor:
    def __init__(self, args, action_dim, obs_dim):
        self.args = args
        self._action_dim = action_dim
        self._obs_dim = obs_dim
        self._dqnet = DQNetwork(args, self._action_dim, obs_dim).to(device)
        self._dqnet_target = DQNetwork(args, self._action_dim, obs_dim).to(device)
        self._dqnet_target.eval()
        self._replay_memory = PrioritizedExpReplay(self.args)#ReplayMemory(args)
        self._optimizer = optim.Adam(self._dqnet.parameters(), lr=self.args.lr, eps=1e-4)
        
        # Compute the learning rate gamma to decay after to min_rl after epsilon_decay updates  
        lr_gamma = (self.args.min_lr / self.args.lr) ** (1/self.args.epsilon_decay)
        print("lr_gamma", lr_gamma)
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self._optimizer, lr_lambda=lambda e: lr_gamma)
        
        self._epsilon = args.epsilon
        self._loss_fn = nn.SmoothL1Loss()
        self._num_steps = 0
        if self.args.load:
            self.load()
      

    def __call__(self, x):
        with torch.no_grad():
            return self.get_action(self._dqnet(x))
    
    def get_action(self, q_values, argmax=False):
        q_values = torch.squeeze(q_values)
        if not argmax and self.epsilon_threshold() >= np.random.rand():
            # Perform random action
            action = np.random.randint(self._action_dim)
        else:
            with torch.no_grad():
                # Perfrom action that maximizes expected return
                action =  q_values.max(0)[1]
        action = np.random.randint(self._action_dim)
        action = int(action)
        return action, q_values[action]
    
    def train(self):
        """Train Q-Network over batch of sampled experience."""
        # Get sample of experience
        is_ws, exs, indices = self._replay_memory.sample(self.args.batch_size)
        #exs = self._replay_memory.sample()
    
        td_targets = torch.zeros(self.args.batch_size).cuda()
        states = torch.zeros(self.args.batch_size, self._obs_dim ).cuda()
        next_states = torch.zeros(self.args.batch_size, self._obs_dim ).cuda()
        rewards = torch.zeros(self.args.batch_size).cuda()
        next_state_mask = torch.zeros(self.args.batch_size).cuda()
        actions = []
        # Create state-action values
        for i, e_t in enumerate(exs):
            states[i] = e_t.state
            actions.append(e_t.action)
            rewards[i] = e_t.reward
            if e_t.next_state is not None:
                next_states[i] = e_t.next_state 
                next_state_mask[i] = 1

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        #print("self._dqnet(states)", self._dqnet(states))
        #print("q_values",self._dqnet(states))
        q_values = self._dqnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next  = self._dqnet(next_states)
            q_next_target  = self._dqnet_target(next_states).detach()

        for i in range(self.args.batch_size):
            if next_state_mask[i] == 0:
                td_targets[i] = rewards[i]
            else:
                # Get the argmax next action for DQN
                action, _ = self.get_action(q_next[i], True)
                
                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                td_targets[i] = rewards[i] + self.args.gamma * q_next_target[i, action]


        # Train model
        self._optimizer.zero_grad()
        td_errors = q_values - td_targets

        #loss = self._loss_fn(q_values, td_targets)
        loss = torch.mean(td_errors ** 2  *  is_ws)
        loss.backward()

        if self.args.use_grad_norm:
            nn.utils.clip_grad.clip_grad_norm_(self._dqnet.parameters(), self.args.grad_norm)


        self._optimizer.step()
        self._num_steps += 1
        self._replay_memory._num_steps += 1
        self._replay_memory.update_priorities(indices, td_errors.detach())

        # Update target policy
        if (self._num_steps + 1) % self.args.target_update_step == 0:
            print("loss", loss)
            print("epsilon_threshold", self.epsilon_threshold())
            print("q_values", q_values)
            print("td_targets", td_targets)
            print("td_errors", td_errors)
            print("replay_len", self.replay_len())
            print("Learning rate", self._optimizer.param_groups[0]["lr"])
            self._dqnet_target.load_state_dict(self._dqnet.state_dict())
        
        # Decay the learning rate
        if self._optimizer.param_groups[0]["lr"] > self.args.min_lr:
            self._lr_scheduler.step()
        else:
            self._optimizer.param_groups[0]["lr"] = self.args.min_lr

    def add_ex(self, e_t):
        """Add a step of experience."""
        with torch.no_grad():
            _, q_value = self.get_action(self._dqnet(e_t.state))
            if e_t.next_state is not None:
                next_action, _ = self.get_action(self._dqnet(e_t.next_state))
                q_next_target = self._dqnet_target(e_t.next_state)[0, next_action]
                td_target = e_t.reward + self.args.gamma *  q_next_target
            else:
                td_target = e_t.reward

            td_error = td_target - q_value

        self._replay_memory.add(e_t, td_error)


    def replay_len(self):
        return self._replay_memory.cur_cap()
    
    def epsilon_threshold(self):
        return max(self.args.epsilon * (1 - (self._num_steps/self.args.epsilon_decay)), self.args.min_epsilon)
        # return maxself.args.min_epsilon + (self.args.epsilon - self.args.min_epsilon) * \
        #     math.exp(-1. * self._num_steps / self.args.epsilon_decay)

    def save(self):
        model_dict = {
            "DQN" : self._dqnet.state_dict(),
            "optimizer" : self._optimizer.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict()
        }
        
        torch.save(model_dict, os.path.join(self.args.save_dir, self.args.model))
        with open(os.path.join(self.args.save_dir, "model_meta.json"), "w") as f:
            json.dump({"num_steps" : self._num_steps}, f)
        self._replay_memory.save()

    def save_best(self):
        model_dict = {
            "DQN" : self._dqnet.state_dict(),
            "optimizer" : self._optimizer.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict()
        }
        
        torch.save(model_dict, os.path.join(self.args.save_best_dir, self.args.model))
        with open(os.path.join(self.args.save_best_dir, "model_meta.json"), "w") as f:
            json.dump({"num_steps" : self._num_steps}, f)
        self._replay_memory.save_best()

    def load(self):
        model_dict = torch.load(os.path.join(self.args.save_dir, self.args.model))
        self._dqnet.load_state_dict(model_dict["DQN"])
        self._dqnet_target.load_state_dict(model_dict["DQN"])
        self._optimizer.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
        with open(os.path.join(self.args.save_dir, "model_meta.json")) as f:
            d = json.load(f)
            self._num_steps = d["num_steps"]

        self._replay_memory._num_steps = self._num_steps
        self._replay_memory.load()


class DQNetwork(nn.Module):
    def __init__(self, args, action_dim, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim , 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        # Outputs state-action value q(s,a) for every action
        return self.fc6(x)