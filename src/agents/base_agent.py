"""Base agent class for backgammon RL agents."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from ..environment.game_state import Move, Player, GameState


class BaseAgent(ABC):
    """Abstract base class for all backgammon agents."""
    
    def __init__(self, player: Player, name: str = "BaseAgent"):
        """Initialize the agent.
        
        Args:
            player: Which player this agent represents (WHITE or BLACK)
            name: Name of the agent for logging/display
        """
        self.player = player
        self.name = name
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
    @abstractmethod
    def choose_action(self, game_state: GameState, legal_moves: List[List[Move]]) -> List[Move]:
        """Choose an action from the available legal moves.
        
        Args:
            game_state: Current game state
            legal_moves: List of legal move sequences
            
        Returns:
            Selected move sequence
        """
        pass
    
    @abstractmethod
    def update(self, game_state: GameState, action: List[Move], reward: float, 
               next_state: GameState, done: bool):
        """Update the agent based on experience.
        
        Args:
            game_state: Previous game state
            action: Action taken
            reward: Reward received
            next_state: Resulting game state
            done: Whether the game ended
        """
        pass
    
    def game_over(self, winner: Optional[Player]):
        """Called when a game ends.
        
        Args:
            winner: The winning player, or None for a draw
        """
        self.games_played += 1
        if winner == self.player:
            self.wins += 1
        elif winner is not None:
            self.losses += 1
            
    def get_win_rate(self) -> float:
        """Get the current win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def reset_stats(self):
        """Reset game statistics."""
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
    def save(self, filepath: str):
        """Save agent to file. Override in subclasses if needed."""
        pass
        
    def load(self, filepath: str):
        """Load agent from file. Override in subclasses if needed."""
        pass
        
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} ({self.player.name}): {self.wins}/{self.games_played} wins ({self.get_win_rate():.2%})"
