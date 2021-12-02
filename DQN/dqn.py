import torch.nn as nn
import torch

# class DQN(nn.Module):
#     def __init__(self, obs_dim, action_dim):
#         super().__init__()

#         self.activation = nn.ReLU()

#         self.fc1 = nn.Linear(obs_dim , 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 32)
#         self.fc6 = nn.Linear(32, action_dim)
        
#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
#         x = self.activation(self.fc4(x))
#         x = self.activation(self.fc5(x))

#         # Outputs state-action value q(s,a) for every action
#         return self.fc6(x)

class DQN(nn.Module):

    def __init__(self, h, w, outputs): # This will be changed with ALE
        super(DQN, self).__init__()

        self.activation = nn.ReLU()
    
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, outputs)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        return x