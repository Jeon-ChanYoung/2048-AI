from Game2048 import Game2048Env, Game2048GUI
from DQN_model import Agent

env = Game2048Env()
agent = Agent(env)

# 훈련된 모델 파일 경로
model_filename = 'Learning_models/2048DQN_1000.pth'
agent.load_model(model_filename)
app = Game2048GUI(agent)
app.mainloop()
