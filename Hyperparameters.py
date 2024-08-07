class Hyperparameters():
    def __init__(self, map_size):
        # self.map_size = map_size
        #self.RL_load_path = f'./final_weights0.999_x_3.pth'
        
        self.learning_rate = 5e-4
        self.discount_factor = 0.6
        self.num_uavs = 3
        self.num_users = 15
        self.area_size = map_size
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes =6000
        self.num_test_episodes = 10
        self.epsilon_decay = 0.9991
        self.buffer_size = 10000
        self.max_steps_per_episode = 60
        self.update_users = 30
        self.save_path = f'./New{self.num_uavs}UAV/Random{self.update_users}final_weights'
        self.RL_load_path = f'./New{self.num_uavs}UAV/Random{self.update_users}final_weights{self.epsilon_decay}_x_{self.num_uavs}.pth'
    def change(self, map_size, batch_size = 32, learning_rate = 5e-4, num_episodes = 3000, epsilon_decay = 0.999):
        '''
        This method can change
        map_size, 
        Also can change the following argument if called:
        batch_size , learning_rate , num_episodes
        '''
        self.map_size = map_size
        self.batch_size = batch_size 
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.epsilon_decay = epsilon_decay
