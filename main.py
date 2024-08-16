from Game2048 import Game2048
from DQN_model import Agent

def train(agent:Agent, game:Game2048, n_games=500):
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
        game.__init__()  # Reset the game
        print(f'| Game: {i} | Score: {score:.2f} | Epsilon: {agent.epsilon:.2f} |')

    return scores

# 에이전트 생성
agent = Agent(gamma=0.99, 
              epsilon=1.0, 
              lr=0.001, 
              input_dims=(16,), 
              batch_size=64, 
              n_actions=4,
              eps_dec=0.0001)

# 게임 환경 생성
game = Game2048()

# 학습 진행
train(agent, game)
