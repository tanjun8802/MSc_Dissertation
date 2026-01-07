import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()
print(obs)
env.close()
print("Test file executed successfully.")