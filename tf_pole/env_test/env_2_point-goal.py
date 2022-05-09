import gym
import safe_rl
import safety_gym
import time


env = gym.make("Safexp-PointGoal1-v0")   # CartPole-v1
observation = env.reset()
start = time.time()
for _ in range(30000):
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)   # swb：info有{'cost_hazards', 'cost'}啊,一个step=0.02s啊！！
    if done:
        observation = env.reset()
end = time.time()
env.close()
duration = end - start
print(f"duration is: {duration}")


