from Game2048 import Game2048
from DQN_model import Agent

def train(agent:Agent, game:Game2048, n_games=100):
    scores = []
    for i in range(n_games):
        done = False
        state = game.get_state()
        score = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done = game.step(action)
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            score += reward
        
        scores.append(score)
        game.__init__() 
        print(f'| Game: {i} | Score: {score:.2f} | Epsilon: {agent.epsilon:.2f} |')

    return scores


gamma = 0.99 # 할인율 (미래 보상에 대한 중요도)
epsilon = 1.0 # 앱실론
lr = 0.001 # 학습률
input_dims = (1, 4, 4)  # 2048 게임 보드의 상태 크기 (채널, 높이, 너비)
batch_size = 64
n_actions = 4 # 가능한 행동의 수 (2048 게임에서 상, 하, 좌, 우)
max_mem_size = 100000 # 리플레이 메모리 최대 크기
epsilon_min = 0.01 # 탐험률 최소값
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

# 게임 환경 생성
game = Game2048()

# 학습 진행
train(agent, game)
agent.save_model('2048dqn_model.pth')
