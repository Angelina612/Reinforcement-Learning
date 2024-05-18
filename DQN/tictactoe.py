import random
import numpy as np

# Defining game environment
class TicTacToe:
    def __init__(self, state=None):
        if state is None:
            self.board = np.zeros((3, 3)).flatten() # initial state of the board
        else:
            self.board = np.array(state).flatten()

        self.players = ['X', 'O']
        self.current_player = 'X'  # Assuming 'X' starts the game
        self.winner = None
        self.game_over = False
    
    def reset(self):
        self.board = np.zeros((3, 3)).flatten()
        self.current_player = None
        self.winner = None
        self.game_over = False
    
    # Valid moves available, returns the positions of empty cells
    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[3*i+j] == 0:
                    moves.append((i, j))
        return moves
    
    def make_move(self, move):
        pos = 3 * move[0] + move[1]
        if self.board[pos] != 0:
            return False
        self.board[pos] = self.players.index(self.current_player) + 1
        self.check_winner()
        self.switch_player()
        return True
    
    # Function for swithching players by turn
    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]
    
    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[3*i] == self.board[3*i+1] == self.board[3*i+2] != 0:
                self.winner = self.players[int(self.board[3*i] - 1)]
                self.game_over = True
        # Check columns
        for j in range(3):
            if self.board[j] == self.board[j+3] == self.board[j+6] != 0:
                self.winner = self.players[int(self.board[j] - 1)]
                self.game_over = True
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.players[int(self.board[0] - 1)]
            self.game_over = True
        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.players[int(self.board[2] - 1)]
            self.game_over = True
        # Check draw
        if not self.available_moves():
            self.game_over = True
        
    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=' ')
            for j in range(3):
                pos = 3*i+j
                print(self.players[int(self.board[pos] - 1)] if self.board[pos] != 0 else " ", end=' | ')
            print()
            print("-------------")

# Function to do a random move
# Used for opponent player
def rand_move(env):
    available_moves = env.available_moves()
    if(env.make_move(random.choice(available_moves))):
        return env.board
    return None