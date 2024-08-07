
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from gym import spaces
import time

class UAVVisualization:
    def __init__(self, num_uavs=5, num_users=25,num_actions_per_uav=5, area_size=(20, 20), max_users_per_uav = 4, plotting_enabled = True, update_users=True):
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.area_size = area_size
        self.max_users_per_uav = max_users_per_uav
        self.step_count = 0
        self.last_update_time = time.time()
        self.num_actions_per_uav = num_actions_per_uav
        self.plotting_enabled = plotting_enabled
        self.update_users = update_users
        cmap = plt.get_cmap('viridis')
        self.colors = cmap(np.linspace(0, 1, num_uavs))
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([num_actions_per_uav] * num_uavs)

        obs_high = np.array([self.area_size[0], self.area_size[1], self.area_size[0], self.area_size[1]] * num_users)
        obs_low = np.zeros_like(obs_high)  # Assuming all coordinates are positive within the defined area size
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Initialize user positions randomly
        #self.user_positions = np.random.rand(self.num_users, 2) * self.area_size
        self.initial_user_positions = np.random.rand(self.num_users, 2) * self.area_size
        self.user_positions = np.copy(self.initial_user_positions)
        # Initialize UAV positions randomly and make sure the distance between UAV is at least 2
        #self.uav_positions = np.random.rand(self.num_uavs, 2) * self.area_size
        self.uav_positions = self.place_uavs_custom()  # Initialize UAV positions array
        self.initia_uav_positions = np.copy(self.uav_positions)
        # Initialize User mover direction and speed
        self.user_direction = np.random.uniform(low = 0, high=2*np.pi, size = self.num_users)
        #self.initia_user_velocities = np.random.uniform(low = 0.1, high = 3, size = self.num_users)
        self.user_speeds = np.random.uniform(low = 0.1, high = 2, size = self.num_users)
        self.uav_user_connections = self.initial_connections()
        self.penalty = 0
        self.setup_plot()
    def calculate_distances(self):
        """Calculate distances from each user to each UAV."""
        # Reshape UAV positions for broadcasting: [1, num_uavs, 2]
        uav_positions_expanded = np.expand_dims(self.uav_positions, axis=0)
        # Reshape user positions for broadcasting: [num_users, 1, 2]
        user_positions_expanded = np.expand_dims(self.user_positions, axis=1)
        # Compute distances using broadcasting, resulting in a [num_users, num_uavs] matrix
        distances = np.linalg.norm(user_positions_expanded - uav_positions_expanded, axis=2)
        
        return distances
    def initial_connections(self):
        """ Establish initial connections based on the closest UAV. """
        connections = {}
        distances = self.calculate_distances()

        # Find the index of the closest UAV for each user
        closest_uavs = np.argmin(distances, axis=1)

        # Initialize connection lists for each UAV
        for uav_index in range(self.num_uavs):
            connections[uav_index] = []

        # Assign each user to their closest UAV
        for user_index, uav_index in enumerate(closest_uavs):
            connections[uav_index].append(user_index)

        return connections
    def place_uavs_custom(self):
        """
        Places UAVs in specific fractions of the area based on the number of UAVs.

        :param num_uavs: Number of UAVs
        :param area_size: (width, height) of the area
        :return: Positions of the UAVs
        """
        self.uav_positions = []
        width, height = self.area_size[0],self.area_size[1]

        if self.num_uavs == 1:
            # Place one UAV in the center
            self.uav_positions.append((width / 2, height / 2))
        elif self.num_uavs == 2:
            # Place two UAVs on the diagonal at 1/4 and 3/4 positions
            self.uav_positions.append((width / 4, height / 4))
            self.uav_positions.append((3 * width / 4, 3 * height / 4))
        elif self.num_uavs == 3:
            # Place two UAVs at 1/4 and 3/4 along the x-axis and one at the center
            self.uav_positions.append((width / 4, height / 3))
            self.uav_positions.append((3 * width / 4, height / 3))
            self.uav_positions.append((width / 2, 2 * height / 3))
        elif self.num_uavs == 4:
            # Place UAVs at the corners of the center rectangle
            self.uav_positions.append((width / 4, height / 4))
            self.uav_positions.append((3 * width / 4, height / 4))
            self.uav_positions.append((width / 4, 3 * height / 4))
            self.uav_positions.append((3 * width / 4, 3 * height / 4))
        elif self.num_uavs == 5:
            # Place four as before plus one in the middle
            self.uav_positions.append((width / 4, height / 4))
            self.uav_positions.append((3 * width / 4, height / 4))
            self.uav_positions.append((width / 4, 3 * height / 4))
            self.uav_positions.append((3 * width / 4, 3 * height / 4))
            self.uav_positions.append((width / 2, height / 2))
        self.uav_positions = np.array(self.uav_positions)
        return self.uav_positions
    def update_connections(self):
        """ Update connections based on current positions or other logic. """
        connections = self.initial_connections()
        return connections
    def update_user_positions(self):
        for i in range(self.num_users):
            dx = self.user_speeds[i] * np.cos(self.user_direction[i])
            dy = self.user_speeds[i] * np.sin(self.user_direction[i])
            new_x = self.user_positions[i][0] + dx
            new_y = self.user_positions[i][1] + dy
            
            if not (0 <= new_x <= self.area_size[0]):
                self.user_direction[i] = np.random.uniform(0, 2 * np.pi)  # Random new angle
                self.user_speeds[i] = np.random.uniform(1, 3)  # Random new speed
                new_x = max(0, min(new_x, self.area_size[0]))

            if not (0 <= new_y <= self.area_size[1]):
                self.user_direction[i] = np.random.uniform(0, 2 * np.pi)  # Random new angle
                self.user_speeds[i] = np.random.uniform(1, 3)  # Random new speed
                new_y = max(0, min(new_y, self.area_size[1]))
            
            self.user_positions[i] = [new_x, new_y]
        return self.user_positions
    def reset(self):
        """Reset the environment to the initial state."""
        # Reset user positions to the initial positions
        self.user_positions = np.copy(self.initial_user_positions)
        self.user_degree = np.random.uniform(low = 0, high = 2*np.pi, size = self.num_users)
        self.user_velocities = np.random.uniform(low = 0.1, high = 3, size = self.num_users)
        # Reset UAV positions
        self.uav_positions = np.copy(self.initia_uav_positions)
        self.uav_user_connections = self.initial_connections()
        self.step_count = 0
        self.penalty = 0
        self.last_update_time = time.time()
        if self.plotting_enabled:
            self.update_plot()  # Optionally update the plot to reflect the reset state
        return self.get_obs()
 
    def step(self, actions):
        """Execute one time step within the environment."""
        # Process UAV movement actions
        self.step_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        movement_map = {
            0: (0, 0),    # Stay
            1: (1, 0),    # Move right
            2: (-1, 0),   # Move left
            3: (0, 1),    # Move up
            4: (0, -1)    # Move down
        }

        # for i, action in enumerate(actions[:self.num_uavs]):
        #     if action > 0:      
        #         dx, dy = movement_map[action]
        #         new_position = self.uav_positions[i] +np .array([dx, dy])
        #         if (0 <= new_position[0] <= self.area_size[0]) and (0 <= new_position[1] <= self.area_size[1]):
        #             self.uav_positions[i] = new_position
        new_positions = np.copy(self.uav_positions)
        for i, action in enumerate(actions[:self.num_uavs]):
            dx, dy = movement_map[action]
            new_position = self.uav_positions[i] + np.array([dx, dy])

            # Check boundaries
            if (5 <= new_position[0] <= self.area_size[0] - 5) and (5 <= new_position[1] <= self.area_size[1] - 5):
                # Temporarily move to check for collisions
                new_positions[i] = new_position
            else: 
                self.penalty = -6
        # Collision checking
        for i in range(self.num_uavs):
            collision = False
            for j in range(self.num_uavs):
                if i != j and np.linalg.norm(new_positions[i] - new_positions[j]) < 3:
                    collision = True
                    break
            if not collision:
                self.uav_positions[i] = new_positions[i]
        if isinstance(self.update_users, bool):
            if self.update_users and elapsed_time >= 60:  # Update every minute if true
                self.user_positions = self.update_user_positions()
                self.update_connections()
                self.last_update_time = current_time
        elif isinstance(self.update_users, int):  # Update every 'update_users' seconds
            if elapsed_time >= self.update_users:
                #self.user_positions = np.random.rand(self.num_users, 2) * self.area_size
                self.user_positions = self.update_user_positions()
                self.update_connections()
                self.last_update_time = current_time

        self.uav_user_connections = self.initial_connections()
        self.update_plot()
        done = False
        rewards = self.calculate_reward()+ self.penalty

        info = {'Step': self.step_count}
        return self.get_obs(),rewards, done, info

    def calculate_reward(self):
        total_reward = 0
        #step_penalty = -3  # Penalty for each step taken
        # edge_penalty = -5  # Penalty for being too close to the edge
        # buffer_zone = 5  # Distance from edge considered too close
        average_dis = self.area_size[0] / (self.num_uavs * 2)
        no_user_penalty = -10  # Increase the penalty for no connections if needed

        for uav_index, users in self.uav_user_connections.items():
            # uav_position = self.uav_positions[uav_index]
            # if (uav_position[0] <= buffer_zone or uav_position[0] >= self.area_size[0] - buffer_zone or
            #     uav_position[1] <= buffer_zone or uav_position[1] >= self.area_size[1] - buffer_zone):
            #     total_reward += edge_penalty
            if len(users) == 0:
                # Apply penalty for no connections
                total_reward += no_user_penalty
            else:
                distances = self.calculate_uav_user_distances(uav_index, users)
                for distance in distances:
                    distance = distance - average_dis
                    normalized_distance = distance / self.area_size[0]
                    # Applying a negative reward proportional to the distance
                    reward = -normalized_distance
                    # reward = -distance
                    total_reward += reward

        return total_reward
    # def calculate_edge_penalty(self, uav_position):
    #     buffer_zone = 5  # How close to the edge to start penalizing
    #     penalty = 0
    #     if uav_position[0] < buffer_zone or uav_position[0] > self.area_size[0] - buffer_zone:
    #         penalty -= 10  # Penalty amount
    #     if uav_position[1] < buffer_zone or uav_position[1] > self.area_size[1] - buffer_zone:
    #         penalty -= 10
    #     return penalty
    def calculate_uav_user_distances(self, uav_index, users):
        """ Calculate distances from a specified UAV to a list of connected users. """
        uav_position = self.uav_positions[uav_index]
        user_positions = self.user_positions[users]
        return np.linalg.norm(user_positions - uav_position, axis=1)

    def get_obs(self):
        # Initialize the observation array
        observation = np.zeros((self.num_users, 4))
        
        for user_index in range(self.num_users):
            # Find the connected UAV index
            for uav_index, users in self.uav_user_connections.items():
                if user_index in users:
                    connected_uav_index = uav_index
                    break
            else:
                # Default to UAV index 0 if no connection (should handle this case properly)
                connected_uav_index = 0
            
            # Fill the observation for this user
            observation[user_index, :2] = self.user_positions[user_index]
            observation[user_index, 2:] = self.uav_positions[connected_uav_index]
        
        return observation
    def setup_plot(self):
        """Setup the initial plotting of UAVs and users."""
        if not self.plotting_enabled:
            return
        self.fig, self.ax = plt.subplots()
        self.uav_scatters = []  # List to store scatter plots for UAVs
        for i in range(self.num_uavs):
            # Create a scatter plot for each UAV with a unique star marker
            # scatter = self.ax.scatter(
            #     self.uav_positions[i, 0], self.uav_positions[i, 1],
            #     color=self.colors[i],  # Use indexing to access colors
            #     marker='*', s=100, label=f'UAV {i+1}'
            # )
            self.ax.scatter(
                self.uav_positions[i, 0], self.uav_positions[i, 1],
                color=self.colors[i], alpha=0.2,  # semi-transparent
                marker='*', s=300  # larger scatter for glow effect
            )
            # Main UAV scatter
            scatter = self.ax.scatter(
                self.uav_positions[i, 0], self.uav_positions[i, 1],
                color=self.colors[i], edgecolor='black',  # distinct edge color
                marker='*', s=200, label=f'UAV {i+1}'  # larger than normal
            )
            self.uav_scatters.append(scatter)  # Store the scatter plot

        # Initialize scatter plots for users and color them based on their connected UAV
        self.user_scatter = []
        for i in range(self.num_users):
            # Find which UAV this user is connected to
            uav_index = None
            for uav, users in self.uav_user_connections.items():
                if i in users:
                    uav_index = uav
                    break

            # If a connection exists, use the UAV's color, otherwise default to grey
            color = self.colors[uav_index] if uav_index is not None else 'grey'
            scatter = self.ax.scatter(
                self.user_positions[i, 0], self.user_positions[i, 1],
                color=color, s=50
            )
            self.user_scatter.append(scatter)  # Store the scatter plot

        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        plt.legend()
        plt.ion()  # Turn on interactive mode to allow live updates
         # Call the update method to start the animation
        plt.show()
        #plt.show()

    def update_plot(self):
        """Update the plot based on new positions."""
        # Update UAV positions
        if not self.plotting_enabled:
            return
        for scatter, new_pos in zip(self.uav_scatters, self.uav_positions):
            scatter.set_offsets(new_pos)

        for user_index, scatter in enumerate(self.user_scatter):
            # Get the index of the UAV to which this user is connected
            uav_index = None
            for uav, users in self.uav_user_connections.items():
                if user_index in users:
                    uav_index = uav
                    break

            # Update the color and position of the user's scatter plot
            if uav_index is not None:
                color = self.colors[uav_index]  # Retrieve the color associated with the connected UAV
                scatter.set_color(color)  # Set the new color

            scatter.set_offsets(self.user_positions[user_index])  # Update the position

        #plt.draw()
        plt.pause(1)  
    def render(self):
        # Simply ensure GUI updates, animation is handled by FuncAnimation
        if self.plotting_enabled:
            plt.pause(5)

def main():
    # Initialize the UAV visualization environment
    env = UAVVisualization(num_uavs=3, num_users=20, area_size=(100, 100))

    # Number of steps to simulate
    #num_steps = 100
    #num_inputs, num_uavs, actions_per_uav
    #actions = [torch.argmax(q_values).item() for q_values in action_values]
 
    observation_space_shape = env.observation_space.shape

    # # Randomly generate some actions for the UAVs and users
    # # Here, actions are randomly chosen from the available action space
    # for _ in range(num_steps):
    #     uav_actions = np.random.randint(0, 5, env.num_uavs)  # Random direction or no movement for each UAV
    #     #user_actions = np.random.randint(0, env.num_uavs, env.num_users)  # Each user randomly selects a UAV to connect to

    #     #actions = np.concatenate((uav_actions, user_actions))
    #     env.step(uav_actions)

    # # Final rendering to show the end state (optional, as the environment already updates in real-time)
    # env.render()

if __name__ == "__main__":
    main()