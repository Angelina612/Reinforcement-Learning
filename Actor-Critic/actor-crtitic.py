# 2001CS06
# Angelina Shibu

import torch
import torch.nn as nn
import torch.optim as optim
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

# Function to do a random move
# Used for opponent player
def rand_move(env):
    available_moves = env.available_moves()
    if(env.make_move(random.choice(available_moves))):
        return env.board
    return None

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Actor head
        self.fc_actor = nn.Linear(hidden_size, 9)

        # Critic head
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        actor_output = self.fc_actor(x)
        critic_output = self.fc_critic(x)

        return actor_output, critic_output

class ActorCriticAgent:
    def __init__(self, input_size, hidden_size, alpha, discount_factor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic_network = ActorCriticNetwork(input_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic_network.parameters(), lr=alpha)
        self.discount_factor = discount_factor

    def choose_action(self, state, available_moves):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actor_output, _ = self.actor_critic_network(state_tensor)
        action_probabilities = torch.softmax(actor_output, dim=1).detach().cpu().numpy().flatten()
        action = tuple(np.unravel_index(np.argmax(action_probabilities), (3, 3)))

        while action not in available_moves:
            action_probabilities[action[0] * 3 + action[1]] = 0.0
            action = tuple(np.unravel_index(np.argmax(action_probabilities), (3, 3)))

        return action

    def update_network(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action[0] * 3 + action[1]]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Calculate TD error
        _, critic_output = self.actor_critic_network(state_tensor)
        _, next_critic_output = self.actor_critic_network(next_state_tensor)
        td_error = reward_tensor + self.discount_factor * next_critic_output - critic_output

        # Actor loss
        actor_output, _ = self.actor_critic_network(state_tensor)
        chosen_action_prob = torch.softmax(actor_output, dim=1).gather(1, action_tensor.view(-1, 1))
        actor_loss = -torch.log(chosen_action_prob) * td_error.detach()

        # Critic loss
        critic_loss = nn.MSELoss()(critic_output, reward_tensor + self.discount_factor * next_critic_output.detach())

        # Total loss
        total_loss = actor_loss + critic_loss

        # Update the network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_reward(self, env):
        if not env.game_over or env.winner == None:
            return 0
        if env.winner == env.current_player: # opponent wins
            return -1
        return 1


# Learning rate: alpha
# Exploration rate: epsilon
def train(num_episodes, alpha, discount_factor):
    input_size = 9
    hidden_size = 32
    agent = ActorCriticAgent(input_size, hidden_size, alpha, discount_factor)

    for i in range(num_episodes):
        if i % (num_episodes/100) == 0:
            print(f"-", end='')

        env = TicTacToe()
        state = env.board

        first_player = random.choice([0, 1])

        # Starting the game
        if first_player == 1: # if first player is not the agent
            state = rand_move(env)

        while not env.game_over:

            # Agent's turn
            available_moves = env.available_moves()
            action = agent.choose_action(state, available_moves)
            
            if env.make_move(action):
                next_state = env.board
                reward = agent.get_reward(env)
                
            agent.update_network(state, action, reward, next_state)
            # agent.update_Q_value(state, action, reward, next_state)
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
    parameters = [(0.8, 1.0),
                  (0.4, 0.8),
                  (0.6, 0.7),
                  (0.5, 1.0)]

    for alpha, discount_factor in parameters:
        print(f"Training for learning rate = {alpha}, discount factor = {discount_factor}")
        # Train the agent
        agent = train(num_episodes=1000, alpha=alpha, discount_factor=discount_factor)

        # Test the agent
        win_percentage = test(agent, num_games=2000)
        print(f"\nVictory percentage: {round(win_percentage, 2)}% \n")