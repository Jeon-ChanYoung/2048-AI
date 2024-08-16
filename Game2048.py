import numpy as np

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
        return self.board.flatten()

    def get_score(self):
        return np.max(self.board)

    def print_board(self):
        print(self.board)

    def step(self, action):
        old_board = self.board.copy()
        
        # Execute the action
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()
        
        # Calculate reward
        if np.array_equal(self.board, old_board):
            # If the board hasn't changed, it means the move was illegal
            reward = -5
        else:
            max_tile_diff = np.max(self.board) - np.max(old_board)
            empty_spaces_diff = np.sum(self.board == 0) - np.sum(old_board == 0)
            reward = max_tile_diff + np.max(self.board) * 0.1 - empty_spaces_diff * 0.1
        
        # Check if the game is done and apply penalty if so
        done = self.is_done()
        if done:
            reward -= 100
        
        return self.get_state(), reward, done
