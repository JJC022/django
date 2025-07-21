"""Random agent for backgammon - baseline for comparison."""

import random
from typing import List
from .base_agent import BaseAgent
from ..environment.game_state import Move, Player, GameState


class RandomAgent(BaseAgent):
    """Agent that chooses moves randomly from legal options."""
    
    def __init__(self, player: Player, name: str = "RandomAgent"):
        """Initialize the random agent."""
        super().__init__(player, name)
        
    def choose_action(self, game_state: GameState, legal_moves: List[List[Move]]) -> List[Move]:
        """Choose a random legal move sequence."""
        if not legal_moves:
            return []
        return random.choice(legal_moves)
    
    def update(self, game_state: GameState, action: List[Move], reward: float, 
               next_state: GameState, done: bool):
        """Random agent doesn't learn, so no update needed."""
        pass
