import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import deque
import torch.nn.functional as F
import Environmentrytryconn as Env

from Hyperparameters import Hyperparameters


######### YOU DON'T NEED TO MODIFY THIS CLASS -- IT'S ALREADY COMPLETED ###############
class Replay_Buffer():
    """
    Experience Replay Buffer to store experiences
    """
    def __init__(self, size, device):

        self.device = device
        self.size = size # size of the buffer
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)
        
        
    def store(self, state, action, next_state, reward, terminal):
        """
        Store experiences to their respective queues
        """      
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        
        
    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, terminals
    
    
    def __len__(self):
        return len(self.terminals)
    
########################################################

################### START FROM HERE ################################

class DQN(nn.Module):
    def __init__(self, num_inputs, num_uavs, actions_per_uav):
        super(DQN, self).__init__()
        self.num_uavs = num_uavs
        self.actions_per_uav = actions_per_uav
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        # Action heads for each UAV
        self.action_heads = nn.ModuleList([
            nn.Linear(128, actions_per_uav) for _ in range(num_uavs)
        ])

    def forward(self, state):
        features = self.shared_layers(state)
        # Output actions for each UAV
        actions = torch.stack([head(features) for head in self.action_heads], dim=0)
        return actions
class Agent:
    """
    Implementing Agent DQL Algorithm
    """
    
    def __init__(self, env:Env, hyperparameters:Hyperparameters, device = False):
        
        # Some Initializations
        if not device:
            if torch.backends.cuda.is_built():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters  

        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0

        #self.action_space  = env.action_space.n
        self.feature_space = env.observation_space.shape[0]
        # self.num_uavs = env.num_uavs
        # self.num_users = env.num_users
        self.action_space = env.num_actions_per_uav
        self.num_uavs = self.hp.num_uavs
        self.num_users = self.hp.num_users
        self.actions_per_uav = env.num_actions_per_uav
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device = self.device)
        
        # Initiate the online and Target DQNs
        ## COMPLETE ##
        self.onlineDQN = DQN(self.feature_space,self.num_uavs,self.actions_per_uav ).to(self.device)
        self.targetDQN = DQN(self.feature_space,self.num_uavs,self.actions_per_uav).to(self.device)

        self.loss_function = nn.MSELoss()

        ## COMPLETE ## 
        # set the optimizer to Adam and call it <self.optimizer>, i.e., self.optimizer = optim.Adam()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)        

    def epsilon_greedy(self, state):
        actions = []
        for uav_index in range(self.num_uavs):
            if np.random.random() < self.epsilon:
                # Select a random action for this UAV
                action = np.random.randint(0, self.actions_per_uav)
            else:
                # Select the action with the highest Q-value for this UAV
                q_values = self.onlineDQN(state)[uav_index]
                action = torch.argmax(q_values).item()
            actions.append(action)
        return actions
    def greedy(self, state):
        """
        Implement greedy policy
        """ 
        # This function should return the action chosen by greedy algorithm # 

        # state = torch.tensor([state], device=self.device, dtype=torch.float32)
        # with torch.no_grad():
        #     q_values = self.onlineDQN(state)
        #     action = torch.argmax(q_values).item()  # Best action from Q-values
        actions = []
        for uav_index in range(self.num_uavs):
            # Select the action with the highest Q-value for this UAV
            q_values = self.onlineDQN(state)[uav_index]
            action = torch.argmax(q_values).item()
            actions.append(action)
        return actions
   

    def apply_SGD(self):
        """
        Train DQN
            ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
            calculate the loss of the episode.
        """ 
        # Sample from the replay buffer
        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)
                    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)       
      
        ## COMPLETE ##
        # Compute <Q_hat> using the online DQN
        all_q_values = self.onlineDQN(states)
        # Compute Q_hat by selecting the Q-values for the actions taken, across each UAV
        q_values_taken_list = []
        for uav in range(self.num_uavs):
            action_indices = actions[:, :,uav].view(-1, 1)
            # Perform gather
            q_values_for_uav = all_q_values[uav].gather(1, action_indices)
            q_values_taken_list.append(q_values_for_uav)
        
        q_values_taken = torch.cat(q_values_taken_list, dim=1)
        with torch.no_grad():   
            ## COMPLETE ##         
            # Compute the maximum Q-value for off-policy update and call it <next_target_q_value> 
            all_next_q_values = self.targetDQN(next_states)
            max_next_q_values = torch.stack([
                all_next_q_values[uav].max(1)[0].unsqueeze(1) for uav in range(self.num_uavs)
            ], dim=1)
           
        # Compute the Q-estimator and call it <y>
        max_next_q_values_summed = max_next_q_values.sum(dim=1)  # This will reduce the size to [32, 1]
        # Compute the Q-estimator
        y = rewards + self.hp.discount_factor * max_next_q_values_summed
        # Update the running loss and learned counts for logging and plotting
        # Sum the Q-values taken for all UAVs at each action
        q_values_taken_summed = q_values_taken.sum(dim=1, keepdim=True)
        # Calculate the loss using these summed values
        loss = self.loss_function(q_values_taken_summed, y)
        self.current_loss += loss.item()
        self.episode_counts += 1

        if self.episode_counts >= self.hp.max_steps_per_episode:
            episode_loss = self.current_loss / self.episode_counts # Average loss per episode
            # Track the loss for final graph
            self.loss_list.append(episode_loss) 
            self.current_loss = 0
            self.episode_counts = 0
        
        # Apply backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip the gradients
        # It's just to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        
        ## COMPLETE ###
        # Update DQN by using the optimizer: <self.optimizer>
        self.optimizer.step()
    ############## THE REMAINING METHODS HAVE BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def update_target(self):
        """
        Update the target network 
        """
        # Copy the online DQN into target DQN
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

    
    def update_epsilon(self):
        """
        reduce epsilon by the decay factor
        """
        # Gradually reduce epsilon
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)
        

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)