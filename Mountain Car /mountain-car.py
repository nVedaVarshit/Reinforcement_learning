import gym
import numpy as np
import pickle

def main():
    # Create the environment
    env = gym.make("MountainCar-v0")

    # Hyperparameters
    num_episodes = 10000  # Number of training episodes
    learning_rate = 0.1  # Alpha: Learning rate
    discount_factor = 0.99  # Gamma: Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    min_epsilon = 0.01  # Minimum exploration rate

    # Discretize state space
    num_bins = (200, 200)  # Number of bins for position and velocity
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bins = [np.linspace(b[0], b[1], num_bins[i] - 1) for i, b in enumerate(state_bounds)]

    # Q-table initialization
    q_table = np.zeros(num_bins + (env.action_space.n,))

    def discretize_state(state):
        """Convert continuous state to discrete indices."""
        return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))

    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment
        state = discretize_state(state)  # Discretize the state
        total_reward = 0

        for _ in range(200):
            # Choose action (epsilon-greedy policy)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Custom reward logic
            position, velocity = next_state
            reward = -2  # Base reward
            if position >= 0.5:
                reward += 100  # Reward for reaching the goal

            if position > 0:
                reward += position  # Reward based on position

            # Reward/penalty based on action and velocity
            if (velocity > 0 and action == 2) or (velocity < 0 and action == 0) or (velocity == 0 and (action == 0 or action == 2)):
                reward += 2  # Reward for moving in the right direction
            else:
                reward -= 10  # Penalty for moving in the wrong direction

            # Apply velocity scaling to the reward
            reward *= abs(velocity)

            # Discretize the next state
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

        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

    # Save the Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

if __name__ == "__main__":
    main()
