import numpy as np
import math
from copy import deepcopy

class Game2048:
    def __init__(self):
        self.board = np.zeros((4,4), dtype=np.int_)
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        y, x = empty_cells[np.random.choice(len(empty_cells))]
        self.board[y, x] = np.random.choice([2, 4], p=[0.9, 0.1])
        return True
    
    def add_tile(self, depth):
        n_empty = len(self.board[self.board == 0])
        if n_empty >= 6 and depth >= 3:
            return self.evaluation(self.board, n_empty)
        if n_empty >= 0 and depth >= 5:
            return self.evaluation(self.board, n_empty)
        if n_empty == 0:
            _, new_score = self.maximize(depth+1)
        sum_score = 0

        empty_cells = list(zip(*np.where(self.board == 0))) #[(행, 열), ... ] 16개
        for y,x in empty_cells:
            for num in [2, 4]:
                new_game = deepcopy(self)  
                new_game.board[y, x] = num
                _, new_score = new_game.maximize(depth+1)

                if num == 2:
                    new_score *= (0.9 / n_empty)
                else:
                    new_score *= (0.1 / n_empty)
                sum_score += new_score
        return sum_score
    
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
        previous_board = self.board.copy()
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()

        changed = not np.array_equal(previous_board, self.board)
        if changed:
            self.add_random_tile()

        n_empty = len(self.board[self.board == 0])
        reward = self.evaluation(self.board, n_empty)
        done = self.is_done()

        return self.get_state(), reward, done

    def evaluation(self, grid, n_empty):
        score = 0

        big_t = np.sum(np.power(grid, 2))

        smoothness = 0
        s_grid = np.sqrt(grid)

        smoothness -= np.sum(np.abs(s_grid[:, 0] - s_grid[:, 1]))
        smoothness -= np.sum(np.abs(s_grid[:, 1] - s_grid[:, 2]))
        smoothness -= np.sum(np.abs(s_grid[:, 2] - s_grid[:, 3]))
        smoothness -= np.sum(np.abs(s_grid[0, :] - s_grid[1, :]))
        smoothness -= np.sum(np.abs(s_grid[1, :] - s_grid[2, :]))
        smoothness -= np.sum(np.abs(s_grid[2, :] - s_grid[3, :]))

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

    def maximize(self, depth=0):
        best_score = -np.inf
        best_action = None

        for action in range(4):
            new_game = deepcopy(self)
            new_game.step(action)
            if np.array_equal(new_game.board, self.board):
                continue

            new_score = new_game.add_tile(depth + 1)
            if best_score <= new_score:
                best_score = new_score
                best_action = action

        return best_action, best_score
