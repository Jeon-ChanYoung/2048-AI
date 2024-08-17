from Game2048 import Game2048GUI
from DQN_model import Agent

gamma = 0.99 
epsilon = 1.0 
lr = 0.001
input_dims = (1, 4, 4)  
batch_size = 64
n_actions = 4
max_mem_size = 100000 
epsilon_min = 0.01 
epsilon_decay = 0.0001
agent = Agent(gamma=gamma, 
              epsilon=epsilon, 
              lr=lr,
              input_dims=input_dims,
              batch_size=batch_size, 
              n_actions=n_actions, 
              max_memory_size=max_mem_size, 
              epsilon_min=epsilon_min,
              epsilon_decay=epsilon_decay)


# 훈련된 모델 파일 경로
model_filename = '2048dqn_model.pth'
agent.load_model(model_filename)
app = Game2048GUI(agent)
app.mainloop()
