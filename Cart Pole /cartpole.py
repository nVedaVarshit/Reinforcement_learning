import gym
import numpy as np
import random
import time

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins, lower_bounds, upper_bounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_bins = num_bins
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.action_space = env.action_space.n
        self.Q = np.random.uniform(low=0, high=1, size=(*num_bins, self.action_space))

    def discretize_state(self, state):
        position, velocity, angle, angular_velocity = state
        cart_position_bins = np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.num_bins[0])
        cart_velocity_bins = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.num_bins[1])
        pole_angle_bins = np.linspace(self.lower_bounds[2], self.upper_bounds[2], self.num_bins[2])
        pole_velocity_bins = np.linspace(self.lower_bounds[3], self.upper_bounds[3], self.num_bins[3])

        position_index = np.digitize(position, cart_position_bins) - 1
        velocity_index = np.digitize(velocity, cart_velocity_bins) - 1
        angle_index = np.digitize(angle, pole_angle_bins) - 1
        angular_velocity_index = np.digitize(angular_velocity, pole_velocity_bins) - 1

        return (position_index, velocity_index, angle_index, angular_velocity_index)

    def choose_action(self, state, episode_idx):
        state_index = self.discretize_state(state)
        if random.random() < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            return np.argmax(self.Q[state_index])

    def update_q_value(self, state, action, reward, next_state, done):
        state_index = self.discretize_state(state)
        next_state_index = self.discretize_state(next_state)
        max_future_q = np.max(self.Q[next_state_index])
        if done:
            target = reward
        else:
            target = reward + self.gamma * max_future_q

        current_q_value = self.Q[state_index + (action,)]
        error = target - current_q_value
        self.Q[state_index + (action,)] += self.alpha * error

    def decay_epsilon(self, episode_idx):
        if episode_idx > 500:
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
    def train(self):
        for episode_idx in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state, episode_idx)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                total_reward += reward
                self.update_q_value(state, action, reward, next_state, done)
                state = next_state

            self.decay_epsilon(episode_idx)

            if (episode_idx + 1) % 100 == 0:
                print(f"Episode {episode_idx + 1}/{self.num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

        print("Training finished!")
        np.save("cartpole_qlearning_improved.npy", self.Q)
        print("Q-table saved successfully!")

    def test_policy(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.choose_action(state, 0)  # Always exploit during testing
            state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
        return total_reward

    def render_learned_policy(self):
        env = gym.make('CartPole-v1', render_mode='human')  # Enable rendering
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.choose_action(state, 0)  # Always exploit learned policy
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.05)  # Control the speed of rendering
        print(f"Total reward during learned strategy: {total_reward}")
        env.close()


def main():
    env = gym.make('CartPole-v1', render_mode=None)  # Disable rendering during training
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    num_episodes = 10000
    num_bins = [10, 10, 10, 10]
    lower_bounds = [-2.4, -2.0, -0.418, -3.5]
    upper_bounds = [2.4, 2.0, 0.418, 3.5]

    agent = QLearningAgent(env, alpha, gamma, epsilon, num_episodes, num_bins, lower_bounds, upper_bounds)
    
    print("Training the agent...")
    agent.train()
    
    print("Testing learned policy without rendering...")
    total_reward = agent.test_policy()
    print(f"Total reward with learned policy: {total_reward}")
    
    print("Rendering learned policy after training...")
    agent.render_learned_policy()  # Render after training

    env.close()


if __name__ == "__main__":
    main()
