import tkinter as tk
from tkinter import messagebox
import numpy as np

"""
Game2048 : 훈련 전용
"""
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int_)
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            y, x = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[y, x] = 2 if np.random.random() < 0.9 else 4

    def move_left(self):
        self.board = np.array([self.merge(row) for row in self.board])
        self.add_tile()

    def move_right(self):
        self.board = np.fliplr(self.board)
        self.move_left()
        self.board = np.fliplr(self.board)

    def move_up(self):
        self.board = np.transpose(self.board)
        self.move_left()
        self.board = np.transpose(self.board)

    def move_down(self):
        self.board = np.transpose(np.fliplr(self.board))
        self.move_left()
        self.board = np.fliplr(np.transpose(self.board))

    def merge(self, row):
        row = row[row != 0]
        merged_row = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(row) and row[i] == row[i + 1]:
                merged_row.append(2 * row[i])
                skip = True
            else:
                merged_row.append(row[i])
        return np.array(merged_row + [0] * (4 - len(merged_row)))

    def is_done(self):
        if not 0 in self.board:
            for move in [self.move_left, self.move_right, self.move_up, self.move_down]:
                copy_board = self.board.copy()
                move()
                if not np.array_equal(self.board, copy_board):
                    self.board = copy_board
                    return False
            return True
        return False
    
    def get_state(self):
        return np.expand_dims(self.board, axis=0)

    def get_score(self):
        return np.max(self.board)

    def print_board(self):
        print(self.board)

    def step(self, action):
        old_board = self.board.copy()
        
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()
        
        if np.array_equal(self.board, old_board):
            reward = -5
        else:
            max_tile_diff = np.max(self.board) - np.max(old_board)
            empty_spaces_diff = np.sum(self.board == 0) - np.sum(old_board == 0)
            reward = max_tile_diff + np.max(self.board) * 0.1 - empty_spaces_diff * 0.1
        
        done = self.is_done()
        if done:
            reward -= 100
        
        return self.get_state(), reward, done

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

