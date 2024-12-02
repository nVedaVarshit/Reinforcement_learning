import gym
import numpy as np
import random

def choose_action(state, Q, epsilon, action_dim):
    # Epsilon-greedy strategy for choosing an action
    if random.random() < epsilon:
        return random.choice(range(action_dim))  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: best action from Q-table

def update_q_table(Q, visited, state, action, reward, next_state, done, learning_rate, gamma):
    # Update Q-value using the Q-learning formula
    current_q_value = Q[state, action]
    
    if done:
        # If the episode ends, no future Q-value
        max_future_q_value = 0
    else:
        # Otherwise, the future Q-value is the max Q-value at the next state
        max_future_q_value = np.max(Q[next_state])

    # Update Q-value using the Bellman equation
    new_q_value = current_q_value + learning_rate * (reward + gamma * max_future_q_value - current_q_value)
    Q[state, action] = new_q_value

    # Track how many times the state-action pair has been visited
    visited[state, action] += 1

def decay_epsilon(visited, epsilon, epsilon_decay, epsilon_min, min_visits_threshold=10):
    # Decay epsilon based on the number of visits
    total_visits = np.sum(visited)
    if total_visits > min_visits_threshold:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return epsilon

def custom_reward(state, done):
    # Custom reward function to encourage better performance
    position, velocity, angle, angular_velocity = state
    
    # Reward the agent for staying upright (angle is close to zero)
    reward = 1.0 - (abs(angle) / 0.2095)  # Reward for angle (keeping pole upright)
    
    # Reward for keeping the position close to the center (i.e., minimizing horizontal movement)
    reward += 1.0 - (abs(position)/2)  # Reward based on position (keeping pole near the center)
    
    # Introduce a penalty for large angular velocity, which means the pole is moving too fast
    reward -= 0.1 * abs(angular_velocity)  # Small penalty for angular velocity
    
    # Introduce a penalty for moving too fast horizontally
    reward -= 0.1 * abs(velocity)  # Small penalty for horizontal velocity
    
    if done:
        # Large penalty if the episode ends prematurely
        reward -= 100  # Penalty if the episode ends
        
    return reward

def main():
    # Initialize environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # Dimension of the state space
    action_dim = env.action_space.n  # Number of possible actions

    # Initialize Q-table and visited table
    Q = np.zeros((state_dim, action_dim))  # Q-table for state-action values
    visited = np.zeros((state_dim, action_dim))  # Track how many times state-action pair was visited

    # Hyperparameters
    learning_rate = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.995  # Epsilon decay factor
    num_episodes = 2000  # Number of episodes to train

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment at the beginning of each episode
        done = False
        total_reward = 0
        
        while not done:
            # Select action based on the current state using epsilon-greedy strategy
            action = choose_action(state, Q, epsilon, action_dim)
            
            # Perform the action and observe the result
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Check if the episode is done
            
            reward = custom_reward(next_state, done)  # Calculate custom reward
            total_reward += reward
            
            # Update the Q-table based on the observed transition
            update_q_table(Q, visited, state, action, reward, next_state, done, learning_rate, gamma)
            
            # Move to the next state
            state = next_state

        # Decay epsilon after each episode based on the number of visits
        epsilon = decay_epsilon(visited, epsilon, epsilon_decay, epsilon_min, min_visits_threshold=500)

        # Print episode stats
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    print("Training finished!")

    # Save the Q-table (optional)
    np.save("cartpole_qlearning.npy", Q)
    print("Q-table saved successfully!")

if __name__ == "__main__":
    main()
