import gym
import torch
import torch.nn as nn
from cart_pole import DQN


env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)

policy_net.load_state_dict(torch.load("cartpole_dqn.pth", weights_only=True))

state, _ = env.reset()  # Unpack the returned tuple
done = False
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = policy_net(state_tensor).argmax().item()
    state, reward, done, _, _ = env.step(action)  # Unpack the additional values returned by step()
    env.render()
env.close()
