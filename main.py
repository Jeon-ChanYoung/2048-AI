from Game2048GUI import Game2048GUI
from algorithms import *

"""
1. Minimax
2. Alpha_Beta_Pruning
3. MCTS
"""

algorithms = {1:Minimax(),
              2:Alpha_Beta_Pruning(),
              3:MCTS()}

app = Game2048GUI(algorithms[3])
app.mainloop()