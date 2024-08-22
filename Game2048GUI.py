from Game2048 import Game2048
from tkinter import messagebox
import tkinter as tk

class Game2048GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2048 Game Simulation")
        self.game = Game2048()
        self.create_widgets()
        self.update_board()
        self.after(1, self.simulate_game)  # Start simulation after 100 ms

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=400, height=400, bg="lightgray")
        self.canvas.pack()

    def simulate_game(self):
        if not self.game.is_done():
            action, _ = self.game.maximize()
            _, _, done = self.game.step(action)
            self.update_board()
            if done:
                messagebox.showinfo("Game Over", f"Game Over! Final Score: {self.game.get_score()}")
                self.quit()  # End the simulation
            else:
                self.after(1, self.simulate_game)  # Continue simulation    

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