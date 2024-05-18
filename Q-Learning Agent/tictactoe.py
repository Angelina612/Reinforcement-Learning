# 2001CS06
# Angelina Shibu

import numpy as np
import random

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

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount_factor):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    # Returns Q-value of a given (state, action)
    def get_Q_value(self, state, action):
        if (tuple(state), action) not in self.Q:
            self.Q[(tuple(state), action)] = 0.0
        return self.Q[(tuple(state), action)]

    # Function to choose the action to be performed
    def choose_action(self, state, available_moves):
        if not available_moves:
            return None  # No available moves in the current state
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            Q_values = [self.get_Q_value(tuple(state), action) for action in available_moves]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def update_Q_value(self, state, action, reward, next_state):
        next_Q_values = [self.get_Q_value(next_state, next_action) for next_action in TicTacToe(next_state).available_moves()]
        max_next_Q = max(next_Q_values) if next_Q_values else 0.0
        self.Q[(tuple(state), action)] = self.get_Q_value(state, action)
        self.Q[(tuple(state), action)] += self.alpha * (reward + self.discount_factor * max_next_Q - self.Q[(tuple(state), action)])
        
    def get_reward(self, env):
        if not env.game_over or env.winner == None:
            return 0
        if env.winner == env.current_player: # opponent wins
            return -1
        return 1

# Function to do a random move
# Used for opponent player
def rand_move(env):
    available_moves = env.available_moves()
    if(env.make_move(random.choice(available_moves))):
        return env.board
    return None

# Learning rate: alpha
# Exploration rate: epsilon
def train(num_episodes, alpha, epsilon, discount_factor):
    agent = QLearningAgent(alpha, epsilon, discount_factor)
    for i in range(num_episodes):
        if i % (num_episodes/100) == 0:
            print(f"-", end='')
        env = TicTacToe()
        state = env.board

        first_player = random.choice([0, 1])

        # Starting the game
        if first_player == 1: # if fisrt player is not the agent
            state = rand_move(env)

        while not env.game_over:

            # Agent's turn
            available_moves = env.available_moves()
            action = agent.choose_action(state, available_moves)
            
            if env.make_move(action):
                next_state = env.board
                reward = agent.get_reward(env)
            agent.update_Q_value(state, action, reward, next_state)
            state = next_state
            
            if env.game_over:
                break
            
            # Opponent's turn
            state = rand_move(env)

    return agent

# Function to test the agent with the given number of games
# Returns the agents win percentage
def test(agent, num_games):
    num_wins = 0
    first_player = random.choice(([0, 1]))
    if first_player == 0:
        ag = 'X'
    else:
        ag = 'O'

    for i in range(num_games):
        game = TicTacToe()
        state = game.board
        
        while not game.game_over:
            if game.current_player == ag:   # Agent's turn
                action = agent.choose_action(state, game.available_moves())
            else:   # Opponent's turn
                action = random.choice(game.available_moves())
            if game.make_move(action):
                reward = agent.get_reward(game)
            if i == 0:
                game.print_board()
        if reward == 1:
            num_wins += 1
    return num_wins / num_games * 100

if __name__ == "__main__":
    # Various parameters to train agent on
    # (learning rate, exploration rate, discount factor)
    parameters = [(0.8, 0.2, 1.0),
                  (0.7, 0.3, 0.8),
                  (0.6, 0.3, 0.7),
                  (0.5, 0.4, 1.0)]

    for alpha, epsilon, discount_factor in parameters:
        print(f"Training for learning rate = {alpha}, exploration rate = {epsilon}, discount factor = {discount_factor}")
        # Train the Q-learning agent
        agent = train(num_episodes=50000, alpha=alpha, epsilon=epsilon, discount_factor=discount_factor)

        # Test the Q-learning agent
        win_percentage = test(agent, num_games=2000)
        print(f"\nVictory percentage: {round(win_percentage, 2)}% \n")