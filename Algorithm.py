import torch
import Environmentrytryconn as env
import numpy as np
import matplotlib.pyplot as plt
from Hyperparameters import Hyperparameters
from AgentPro import Agent

class DQL():
    def __init__(self, hyperparameters:Hyperparameters, train_mode):

        if train_mode:
            plotting_enabled = False
        else:
            plotting_enabled = True

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        # plotting_enabled = True
        self.hp = hyperparameters
        # Load the environment
        self.env = env.UAVVisualization(num_uavs=self.hp.num_uavs, num_users=self.hp.num_users, area_size=self.hp.area_size, plotting_enabled = plotting_enabled, update_users = self.hp.update_users)
        # Initiate the Agent
        self.agent = Agent(env = self.env, hyperparameters = self.hp)
                
    def preprocess(self,state):
        # Normalize the observation to scale the position coordinates
        # Assuming the area size is known and fixed, you can normalize positions by the size of the area.
        max_position_value = self.hp.area_size[0]  # assuming area_size is [100, 100]
        normalized_observation = state / max_position_value
        observation_tensor = torch.FloatTensor(normalized_observation).view(-1) 
        return observation_tensor
    def train(self): 
        """                
        Traing the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in range(1, self.hp.num_episodes+1):
            # sample a new state
            state = self.env.reset()
            # state = self.feature_representation(state)
            state = self.preprocess(state)
            step_size = 0
            episode_reward = 0
                                                
            for _ in range(self.hp.max_steps_per_episode):
                # find <action> via epsilon greedy 
                # use what you implemented in Class Agent
                action = self.agent.epsilon_greedy(state)
                # find nest state and reward
                next_state, reward, ended, _ = self.env.step(action)
                next_state = self.preprocess(next_state)

                # Find the feature of  <next_state> using your implementation <self.feature_representation>
                #next_state = self.feature_representation(next_state)
                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, ended) 
                if len(self.agent.replay_buffer) > self.hp.batch_size:
                    # use <self.agent.apply_SGD> implementation to update the online DQN
                    self.agent.apply_SGD()
                    # Update target-network weights
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        # Copy the online DQN into the Target DQN using what you implemented in Class Agent
                        self.agent.update_target()
                state = next_state
                episode_reward += reward
                step_size +=1
                
            # 
            self.collected_rewards.append(episode_reward)                     
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                      f"Total Time Steps: {total_steps}, "
                      f"Trajectory Length: {step_size}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.2f}")
            print(printout)
        #self.agent.save(self.hp.save_path + '{self.hp.epsilon_decay}'+'.pth')
        # self.env.save_initial_positions()
        
        self.agent.save(f"{self.hp.save_path}{self.hp.epsilon_decay}_x_{self.hp.num_uavs}.pth")

        self.plot_learning_curves()
                                                                    

    def play(self):  
        """                
        play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()
        max_steps_per_episode = 50
        for episode in range(1, self.hp.num_test_episodes+1): 
            state = self.env.reset()
            # print(f"Initial State for Episode {episode}: {state}")
            state = self.preprocess(state)
            ended = False
            step_size = 0
            episode_reward = 0                                              
            #while not ended and not truncated:
            for step in range(max_steps_per_episode):
                # Find the feature of <state> using your implementation <self.feature_representation>
                # Act greedy and find <action> using what you implemented in Class Agent
                
                action = self.agent.greedy(state)
                
                next_state, reward, ended, _ = self.env.step(action)
                # print(next_state)
                # print(f"Stat: {next_state},Step: {step}, Action: {action}, Reward: {reward}")
                next_state = self.preprocess(next_state)
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
            
        #pygame.quit()
        
    ############## THIS METHOD HAS BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(50)/50, mode='valid')
        
        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.ylim([np.min(moving_average)-20, np.max(moving_average)+20])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./New{self.hp.num_uavs}UAV/Reward_vs_Episode_{self.hp.update_users}_x_{self.hp.epsilon_decay}_x_{self.hp.learning_rate}_x_{self.hp.num_uavs}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")
        
       # Save the figure
        plt.savefig(f'./New{self.hp.num_uavs}UAV/Learning_Curve_{self.hp.update_users}_x_{self.hp.epsilon_decay}_x_{self.hp.learning_rate}_x_{self.hp.num_uavs}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()        
        