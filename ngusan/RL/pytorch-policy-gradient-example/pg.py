import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb
from gym import spaces
from wrappers import *


# class PolicyNet(nn.Module):
#     def __init__(self):
#         super(PolicyNet, self).__init__()

#         self.fc1 = nn.Linear(6400 , 256)
#         self.fc2 = nn.Linear( 256, 16)
#         self.fc3 = nn.Linear(16, 1)  # Prob of Left

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.sigmoid(self.fc3(x))
#         return x

class PG(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)
        return self.fc(conv_out)

def main():

    # Plot duration curve: 
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Parameters
    device=torch.device("cpu" )
    num_episode = 100000
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99

    env = gym.make('Pong-v0')
    env.seed(42)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(
        env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    policy_net = PG(env.observation_space,env.action_space).to(device)
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    r = 0
    steps = 0


    for e in range(num_episode):

        state = env.reset()
        state = np.array(state) / 255.0
        r=0
        
        

        for t in count():

            
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            probs = policy_net(state)
            m = Bernoulli(probs)
            action = m.sample()
            
            action = action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            r += reward
            
            env.render(mode='rgb_array')
            
            # To mark boundarys between episodes
            if done:
                
                reward = 0

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            state = np.array(state) / 255.0
            

            steps += 1

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        print(e,r)
        
        # np.savetxt('rewards_per_episode.csv', r,
        #                delimiter=',', fmt='%1.3f')
        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = policy_net(state)
                m = Bernoulli(probs)
                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


if __name__ == '__main__':
    main()
