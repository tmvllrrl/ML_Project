import torch
import gym
from torch import nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import math
import os, json

from dqn import DQN
from replay_memory import Experience, ReplayMemory



class Pipeline:

    def __init__(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = gym.make("Qbert-v0")

        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 100000
        self.target_update = 10
        self.episodes = 10000
        self.save_dir = "best_model"

        init_screen = self.get_screen()
        _, _, self.screen_height, self.screen_width = init_screen.shape

        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)

        # self.policy_net = DQN(128, self.n_actions).to(self.device)
        # self.target_net = DQN(128, self.n_actions).to(self.device)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.eps_threshold = 0

        self.rewards = []
        self.max_reward = 0.0
        self.max_avg_reward = 0.0

    def train(self):
        for i in range(self.episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()

            done = False
            total_reward = 0
            eps_length = 0

            state = self.get_screen().to(self.device)

            while not done:
                self.env.render()

                action = self.select_action(state)
                new_obs, reward, done, _ = self.env.step(action.item())

                if reward > 0:
                    total_reward += reward

                reward = torch.tensor([reward], device=self.device)

                if not done:
                    next_state = self.get_screen().to(self.device)
                    # next_state = torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0).cuda()
                else:
                    next_state = None
                
                self.memory.push(state, action, reward, next_state)

                state = next_state
                eps_length += 1

                self.optimize_model()

            
            self.rewards.append(total_reward)

            if i > 50:
                avg_reward = round(np.mean(self.rewards[-50:]), 2)
            else:
                avg_reward = round(np.mean(self.rewards), 2)
            
            if avg_reward > self.max_avg_reward:
                self.max_avg_reward = avg_reward

                print("Saving Best Model")
                self.save()
            
            if total_reward > self.max_reward:
                self.max_reward = total_reward
            
            print(f"Episode: {i}\t Total Reward: {total_reward}\t Avg Reward: {avg_reward}\t Max Reward: {self.max_reward}\t Max Avg Reward: {self.max_avg_reward}\t Eps Length: {eps_length}")
            print(f"Epsilon Threshold: {self.eps_threshold}")

            if i % self.target_update == 0:
                print("Updating Target Network")
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print("Done Training")
        self.env.close()

    def select_action(self, state):
        sample = random.random()

        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1.0 * self.steps_done / self.eps_decay)
        
        if sample > self.eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1,1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)

        self.steps_done += 1

        return action
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state).cuda()
        action_batch = torch.cat(batch.action).cuda()
        reward_batch = torch.cat(batch.reward).cuda()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)

        self.optimizer.step()

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        
        # Cart is in the lower half, so strip off the top and bottom of the screen
        # _, screen_height, screen_width = screen.shape
        # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        # view_width = int(screen_width * 0.6)
        # cart_location = self.get_cart_location(screen_width)
        # if cart_location < view_width // 2:
        #     slice_range = slice(view_width)
        # elif cart_location > (screen_width - view_width // 2):
        #     slice_range = slice(-view_width, None)
        # else:
        #     slice_range = slice(cart_location - view_width // 2,
        #                         cart_location + view_width // 2)
        # # Strip off the edges, so that we have a square image centered on a cart
        # screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0)
    
    def save(self):
        model_dict = {
            "DQN" : self.policy_net.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }
        
        torch.save(model_dict, os.path.join(self.save_dir, "./model"))
        with open(os.path.join(self.save_dir, "model_meta.json"), "w") as f:
            json.dump({"num_steps" : self.steps_done}, f)
        
        with open(os.path.join(self.save_dir, "ep_reward.json"), "w") as f:
            json.dump(self.rewards, f)

        with open(os.path.join(self.save_dir, "stats.txt"), "w") as f:
            f.write(f"{self.max_avg_reward}\n")
            f.write(f"{self.max_reward}")

if __name__ == "__main__":
    pl = Pipeline()
    pl.train()