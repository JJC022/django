"""TD-Gammon style agent using temporal difference learning."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from .base_agent import BaseAgent
from ..environment.game_state import Move, Player, GameState


class TDNetwork(nn.Module):
    """Neural network for TD-Gammon style value function approximation."""
    
    def __init__(self, input_size: int = 201, hidden_sizes: List[int] = [200, 200]):
        """Initialize the network.
        
        Args:
            input_size: Size of input features (board representation)
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
            
        # Output layer: single value for position evaluation
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Tanh())  # Output between -1 and 1
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class TDAgent(BaseAgent):
    """TD-Gammon style agent using temporal difference learning."""
    
    def __init__(self, player: Player, name: str = "TDAgent", 
                 learning_rate: float = 0.001, epsilon: float = 0.1,
                 lambda_param: float = 0.9, device: str = "cpu"):
        """Initialize the TD agent.
        
        Args:
            player: Which player this agent represents
            name: Name of the agent
            learning_rate: Learning rate for neural network
            epsilon: Exploration rate (epsilon-greedy)
            lambda_param: TD-lambda parameter for eligibility traces
            device: Device to run computations on
        """
        super().__init__(player, name)
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.device = torch.device(device)
        
        # Initialize neural network
        self.network = TDNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training state
        self.training = True
        self.eligibility_traces = {}
        self.previous_state = None
        self.previous_value = None
        
    def choose_action(self, game_state: GameState, legal_moves: List[List[Move]]) -> List[Move]:
        """Choose action using epsilon-greedy policy with value function."""
        if not legal_moves:
            return []
            
        # Epsilon-greedy exploration
        if self.training and np.random.random() < self.epsilon:
            return np.random.choice(legal_moves)
            
        # Evaluate each possible move sequence
        best_move = legal_moves[0]
        best_value = float('-inf')
        
        for move_sequence in legal_moves:
            # Create temporary game state to evaluate move
            temp_state = game_state.copy()
            temp_state.make_move_sequence(move_sequence)
            
            # Get value of resulting position
            value = self.evaluate_position(temp_state)
            
            # Adjust value based on player perspective
            if self.player == Player.BLACK:
                value = -value
                
            if value > best_value:
                best_value = value
                best_move = move_sequence
                
        return best_move
    
    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate a game position using the neural network."""
        features = torch.FloatTensor(game_state.get_state_vector()).to(self.device)
        
        with torch.no_grad():
            value = self.network(features.unsqueeze(0)).item()
            
        return value
    
    def update(self, game_state: GameState, action: List[Move], reward: float, 
               next_state: GameState, done: bool):
        """Update the agent using TD learning."""
        if not self.training:
            return
            
        # Convert states to features
        current_features = torch.FloatTensor(game_state.get_state_vector()).to(self.device)
        
        # Get current value estimate
        current_value = self.network(current_features.unsqueeze(0))
        
        if done:
            # Terminal state - use actual reward
            target_value = torch.FloatTensor([reward]).to(self.device)
        else:
            # Non-terminal state - bootstrap from next state
            next_features = torch.FloatTensor(next_state.get_state_vector()).to(self.device)
            with torch.no_grad():
                next_value = self.network(next_features.unsqueeze(0))
            target_value = reward + 0.99 * next_value  # Discount factor of 0.99
            
        # Compute TD error
        td_error = target_value - current_value
        
        # Update network
        self.optimizer.zero_grad()
        loss = td_error.pow(2)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store for eligibility traces (simplified version)
        self.previous_state = current_features
        self.previous_value = current_value.detach()
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.network.train(training)
        
    def set_epsilon(self, epsilon: float):
        """Set exploration rate."""
        self.epsilon = epsilon
        
    def save(self, filepath: str):
        """Save the agent's neural network."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'player': self.player,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'lambda_param': self.lambda_param,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses
        }, filepath)
        
    def load(self, filepath: str):
        """Load the agent's neural network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.lambda_param = checkpoint.get('lambda_param', self.lambda_param)
        self.games_played = checkpoint.get('games_played', 0)
        self.wins = checkpoint.get('wins', 0)
        self.losses = checkpoint.get('losses', 0)
