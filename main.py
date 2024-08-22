from Game2048GUI import Game2048GUI
from algorithms import *

"""
1. Minimax
2. Alpha_Beta_Pruning
"""

algorithms = {1:Minimax(),
              2:Alpha_Beta_Pruning()}

app = Game2048GUI(algorithms[2])
app.mainloop()