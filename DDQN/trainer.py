import gym
import matplotlib.pyplot as plt 
import cv2, torch
import torch.nn as nn
import numpy as np
import os, json

from dqn import DQNActor
from replay_memory import Experience

DO_NOTHING_ACTION = 1


class Trainer:
    def __init__(self, args, actor, env):
        self.args = args
        self._env = env
        print(type(env))
        self._agent = actor

        if self.args.load:
            self.load_results()
        else:
            self.total_ep_reward = []

        self.rewards = []
        self.max_reward = 0.0
        self.max_avg_reward = 0.0
        
        
    def current_frame(self):
        """Get current frame."""
        # Get frame
        frame = self._env.render(mode='rgb_array')

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize the frame
        frame = cv2.resize(frame, (self.args.img_dim, self.args.img_dim), interpolation = cv2.INTER_LINEAR)
        
        # Normalize frame to [-1, 1]
        frame = frame / 127.5 - 1
        
        return torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)

    def run(self):
        for e_i in range(self.args.episodes):
            ob = self._env.reset() 
            state = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).cuda()
    
            total_reward = 0
            done = False
            num_steps = 0
            while not done:
                self._env.render()
                # Get current action
                action, _ = self._agent(state)
                
                # Perform action in environment
                ob, reward, done, _ = self._env.step(action)
                total_reward += reward

                # Add memory step
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).cuda()
                
                if next_state is None:
                    e_t = Experience(state.clone(), action, reward, next_state)
                else:
                    e_t = Experience(state.clone(), action, reward, next_state.clone())
                
                self._agent.add_ex(e_t)
                state = next_state
                
                if self._agent.replay_len() > self.args.min_init_state and self._agent.replay_len() > self.args.batch_size and (num_steps+1) % self.args.update_steps == 0:
                    self._agent.train()

                num_steps += 1

            self.rewards.append(total_reward)
            if e_i > 25:
                avg_reward = np.mean(self.rewards[-50:])
            else:
                avg_reward = np.mean(self.rewards)
            
            avg_reward = round(avg_reward, 2)

            if total_reward > self.max_reward: 
                self.max_reward = total_reward
            if avg_reward > self.max_avg_reward: 
                self.max_avg_reward = avg_reward
                print("Saving new best model")
                self._agent.save_best()
                self.save_best_results()


            self._env.reset()
            self._agent.save()
            if self._agent.replay_len() > self.args.min_init_state:
                self.total_ep_reward.append(total_reward)
            self._env.reset()
            print(f"Episode: {e_i}\t Total_reward: {total_reward}\t Avg. Reward: {avg_reward}\t Max Reward: {self.max_reward}\t Max Avg Reward: {self.max_avg_reward}\t Episode Length: {num_steps}")
            if e_i % self.args.save_iter == 0:
                self._agent.save()
                self.save_results()

            # for i in range(8):
            #     self._agent.train()
            
        
        self._env.close()

    def load_results(self):
        with open(os.path.join(self.args.save_dir, "ep_reward.json"), "r") as f:
            self.total_ep_reward = json.load(f)

    def save_results(self):
        with open(os.path.join(self.args.save_dir, "ep_reward.json"), "w") as f:
            json.dump(self.total_ep_reward, f)
    
    def save_best_results(self):
        with open(os.path.join(self.args.save_best_dir, "ep_reward.json"), "w") as f:
            json.dump(self.total_ep_reward, f)

        with open(os.path.join(self.args.save_best_dir, "stats.txt"), "w") as f:
            f.write(f"{self.max_avg_reward}\n")
            f.write(f"{self.max_reward}")

