import gym
import safe_rl
import safety_gym
import time


env = gym.make("Acrobot-v1")   # CartPole-v0 max_episode_length=500, reward_threshold=-100,
observation = env.reset()
start = time.time()
for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
end = time.time()
env.close()
duration = end - start
print(f"duration is: {duration}")



