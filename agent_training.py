from Game2048 import *
from DQN_model import *

env = Game2048Env()
agent = Agent(env)
agent.load_model("Learning_models/2048DQN_1000.pth")
num_episodes = 1000
target_update_freq = 10

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    if episode % target_update_freq == 0:
        agent.update_target()
    
    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
agent.save_model("2048DQN_2000.pth")
