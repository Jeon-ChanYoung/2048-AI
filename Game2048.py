import tkinter as tk
from tkinter import messagebox
import numpy as np
import math

"""
Game2048 : 훈련 전용
"""
class Game2048:
    def __init__(self):
        self.board = np.zeros((4,4), dtype=np.int_)
        self.add_tile()
        self.add_tile()
    
    def add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0))) #[(행, 열), ... ] 16개
        if empty_cells:
            y, x = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[y, x] = [2, 4][np.random.random() > 0.9]
    
    def merge(self, row):
        row = row[row != 0]
        merged_row = []
        
        i = 0
        while i < len(row):
            if i+1 < len(row) and row[i] == row[i+1]:
                merged_row.append(2*row[i])
                i += 2
            else:
                merged_row.append(row[i])
                i += 1
        return np.array(merged_row + [0]*(4-len(merged_row)))

    def move_left(self):
        self.board = np.array([self.merge(row) for row in self.board])
    
    def move_right(self):
        self.board = np.fliplr(self.board)
        self.move_left()
        self.board = np.fliplr(self.board)
    
    def move_up(self):
        self.board = np.transpose(self.board)
        self.move_left()
        self.board = np.transpose(self.board)
    
    def move_down(self):
        self.board = np.transpose(self.board)
        self.move_right()
        self.board = np.transpose(self.board)          

    def is_done(self):
        if 0 in self.board:
            return False

        for row in self.board:
            for i in range(len(row) - 1):
                if row[i] == row[i+1]:
                    return False
        
        for col in self.board.T:
            for i in range(len(col) - 1):
                if col[i] == col[i+1]:
                    return False
        return True

    def get_state(self):
        return np.expand_dims(self.board, axis=0)

    def get_score(self):
        return np.max(self.board)

    def step(self, action):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()
        self.add_tile()
        
        n_empty = len(self.board[self.board == 0])
        reward = self.evaluation(self.board, n_empty)
        done = self.is_done()

        return self.get_state(), reward, done

    def evaluation(self, grid, n_empty):
        score = 0

        # grid sum
        big_t = np.sum(np.power(grid, 2))

        # smoothness
        smoothness = 0
        s_grid = np.sqrt(grid)

        smoothness -= np.sum(np.abs(s_grid[:, 0] - s_grid[:, 1]))
        smoothness -= np.sum(np.abs(s_grid[:, 1] - s_grid[:, 2]))
        smoothness -= np.sum(np.abs(s_grid[:, 2] - s_grid[:, 3]))
        smoothness -= np.sum(np.abs(s_grid[0, :] - s_grid[1, :]))
        smoothness -= np.sum(np.abs(s_grid[1, :] - s_grid[2, :]))
        smoothness -= np.sum(np.abs(s_grid[2, :] - s_grid[3, :]))

        # monotonicity
        monotonic_up = 0
        monotonic_down = 0
        monotonic_left = 0
        monotonic_right = 0

        for x in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 3 and not grid[next, x]:
                    next += 1
                current_cell = grid[current, x]
                current_value = math.log(current_cell, 2) if current_cell else 0
                next_cell = grid[next, x]
                next_value = math.log(next_cell, 2) if next_cell else 0
                if current_value > next_value:
                    monotonic_up += (next_value - current_value)
                elif next_value > current_value:
                    monotonic_down += (current_value - next_value)
                current = next
                next += 1

        for y in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 3 and not grid[y, next]:
                    next += 1
                current_cell = grid[y, current]
                current_value = math.log(current_cell, 2) if current_cell else 0
                next_cell = grid[y, next]
                next_value = math.log(next_cell, 2) if next_cell else 0
                if current_value > next_value:
                    monotonic_left += (next_value - current_value)
                elif next_value > current_value:
                    monotonic_right += (current_value - next_value)
                current = next
                next += 1

        monotonic = max(monotonic_up, monotonic_down) + max(monotonic_left, monotonic_right)
        
        # weight for each score
        empty_w = 100000
        smoothness_w = 3
        monotonic_w = 10000

        empty_u = n_empty * empty_w
        smooth_u = smoothness ** smoothness_w
        monotonic_u = monotonic * monotonic_w

        score += big_t
        score += empty_u
        score += smooth_u
        score += monotonic_u

        return score
    
class Game2048Env:
    def __init__(self):
        self.game = Game2048()
        self.action_space = 4  # Left, Right, Up, Down
        self.observation_space = (4, 4)

    def reset(self):
        self.game = Game2048()
        return self.game.get_state()

    def step(self, action):
        state, reward, done = self.game.step(action)
        return state, reward, done, {}

    def render(self):
        print(self.game.board)

"""
Game2048GUI : 시각화 전용
"""

class Game2048GUI(tk.Tk):
    def __init__(self, agent):
        super().__init__()
        self.title("2048 Game Simulation")
        self.agent = agent
        self.game = Game2048()
        self.create_widgets()
        self.update_board()
        self.after(100, self.simulate_game)  # Start simulation after 100 ms

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=400, height=400, bg="lightgray")
        self.canvas.pack()

    def simulate_game(self):
        if not self.game.is_done():
            state = self.game.get_state()
            action = self.agent.choose_action(state)
            _, _, done = self.game.step(action)
            self.update_board()
            if done:
                messagebox.showinfo("Game Over", f"Game Over! Final Score: {self.game.get_score()}")
                self.quit()  # End the simulation
            else:
                self.after(100, self.simulate_game)  # Continue simulation    

    def update_board(self):
        self.canvas.delete("all")
        tile_colors = {0: "lightgray", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
                       32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850",
                       1024: "#edc53f", 2048: "#edc22e"}
        
        for i in range(4):
            for j in range(4):
                value = self.game.board[i, j]
                x1, y1 = j * 100, i * 100
                x2, y2 = x1 + 100, y1 + 100
                color = tile_colors.get(value, "black")
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="white")
                if value != 0:
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(value), font=("Arial", 24, "bold"))

