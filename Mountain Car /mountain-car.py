import gym
import numpy as np

class CustomMountainCarEnv(gym.Wrapper):
    def _init_(self, env):
        super()._init_(env)

    def step(self, action):
        # Perform the default step
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Rescale position (subtract 0.5 to center goal at zero)
        position, velocity = state
        rescaled_position = position - 0.5
        
        # If the agent has passed the goal, give +100 reward and terminate the episode
        if position >= 0.5:
            reward = 100
            terminated = True  # End the episode if the goal is reached
        else:
            # Negative reward based on position, closer to goal = less negative
            reward = -rescaled_position  # Negative reward for distance from goal
            
            # Further penalize low velocity (encouraging high velocity)
            if velocity != 0:
                reward /= abs(velocity)  # Decrease penalty with higher velocity
            else:
                reward = -10  # High penalty if velocity is zero (in case of stuck agent)
        
        return state, reward, terminated, truncated, info

# Main function to train the agent
def main():
    env = CustomMountainCarEnv(gym.make("MountainCar-v0"))

    # Hyperparameters
    num_episodes = 10000
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # Discretize state space
    num_bins = (200, 200)
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]

    # Q-table initialization
    q_table = np.zeros(num_bins + (env.action_space.n,))

    def discretize_state(state):
        """Convert continuous state to discrete indices."""
        return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state)
        total_reward = 0

        for _ in range(200):
            # Choose action (epsilon-greedy policy)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)

            # Update Q-value
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += learning_rate * (
                reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action]
            )

            state = next_state
            total_reward += reward
            if terminated or truncated:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

    # Save the Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

if __name__ == "__main__":
    main()
