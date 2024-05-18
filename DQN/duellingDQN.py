import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tictactoe import TicTacToe, rand_move

# Define the DuelingQNetwork class
class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Advantage stream
        self.fc_advantage = nn.Linear(hidden_size, output_size)

        # Value stream
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        advantage = self.fc_advantage(x)
        value = self.fc_value(x)

        return value + advantage - advantage.mean(dim=0, keepdim=True)
    
# Define the DuelingDQNAgent class
class DuelingDQNAgent:
    def __init__(self, input_size, output_size, hidden_size, alpha, gamma, epsilon):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q_network1 = DuelingQNetwork(input_size, output_size, hidden_size)
        self.Q_network2 = DuelingQNetwork(input_size, output_size, hidden_size)

        self.optimizer1 = optim.Adam(self.Q_network1.parameters(), lr=alpha)
        self.optimizer2 = optim.Adam(self.Q_network2.parameters(), lr=alpha)

        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, available_moves):
        if not available_moves:
            return None  # No available moves in the current state

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Get Q-values for the available moves using Q_network1
        Q_values_main = self.Q_network1(state_tensor).detach().cpu().numpy()

        # Get Q-values for the available moves using Q_network2
        Q_values_target = self.Q_network2(state_tensor).detach().cpu().numpy()

        for move in available_moves:
            row, col = move
            index = row * 3 + col
            Q_values_main[index] += Q_values_target[index]

        best_move_index = np.argmax(Q_values_main)
        best_move = np.unravel_index(best_move_index, (3, 3))

        return best_move


    def update_Q_values(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)

        # Get Q-values from both networks for the current state
        Q1_values = self.Q_network1(state_tensor)

        # Get the minimum Q-value between Q1 and Q2 for the next state
        Q1_next = self.Q_network1(next_state_tensor).detach().cpu()
        Q2_next = self.Q_network2(next_state_tensor).detach().cpu()
        Q_next = torch.min(Q1_next, Q2_next)

        target = reward_tensor + self.gamma * Q_next.max(dim=0).values.unsqueeze(0)

        # Calculate the loss
        loss = nn.MSELoss()(Q1_values, target)

        # Optimize the model
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

        self.Q_network2.load_state_dict(self.Q_network1.state_dict())

    def get_reward(self, env):
        if not env.game_over or env.winner == None:
            return 0
        if env.winner == env.current_player: # opponent wins
            return -1
        return 1 # agent wins


# Learning rate: alpha
# Exploration rate: epsilon
def train(num_episodes, alpha, epsilon, gamma):
    input_size = 9  # Size of the state space (TicTacToe board)
    output_size = 9  # Number of possible actions
    hidden_size = 128  # Size of the hidden layer in the neural network
    gamma = gamma
    agent = DuelingDQNAgent(input_size, output_size, hidden_size, alpha, gamma, epsilon)

    for i in range(num_episodes):
        if i % (num_episodes/100) == 0:
            print(f"-", end='')
        # print(i)
        env = TicTacToe()
        state = env.board

        while not env.game_over:
            available_moves = env.available_moves()

            # Agent's turn
            action = agent.choose_action(state, available_moves)
            # print(action)

            if env.make_move(action):
                next_state = env.board

            # Update Q-values
            agent.update_Q_values(state, action, agent.get_reward(env), next_state)
            # state = next_state

            if env.game_over:
                break

            # Opponent's turn
            state = rand_move(env)

    return agent

# Function to test the agent with the given number of games
# Returns the agents win percentage
def test(agent, num_games):
    num_wins = 0
    for i in range(num_games):
        game = TicTacToe()
        state = game.board
        done = False

        while not done:
            if game.current_player == 'X':
                action = agent.choose_action(state, game.available_moves())
            else:
                action = random.choice(game.available_moves())

            if game.make_move(action):
                reward = agent.get_reward(game)
                done = game.game_over

            if i == 0:
                game.print_board()

            if done:
                break

            state = game.board

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

    for alpha, epsilon, gamma in parameters:
        print(f"Training for learning rate = {alpha}, exploration rate = {epsilon}, discount factor = {gamma}")
        # Train the Q-learning agent
        agent = train(num_episodes=10000, alpha=alpha, epsilon=epsilon, gamma=gamma)

        # Test the Q-learning agent
        win_percentage = test(agent, num_games=2000)
        print(f"\nVictory percentage: {round(win_percentage, 2)}% \n")

