"""Evaluation metrics for backgammon agents."""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from ..environment.game_state import GameState, Player


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    winner: Player
    game_length: int
    white_moves: int
    black_moves: int
    white_checkers_hit: int
    black_checkers_hit: int
    white_bear_off_moves: int
    black_bear_off_moves: int
    game_duration: float


class EvaluationMetrics:
    """Comprehensive evaluation metrics for backgammon agents."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.game_metrics = []
        
    def record_game(self, game_state: GameState, game_history: List[Dict], 
                   duration: float) -> GameMetrics:
        """Record metrics for a completed game.
        
        Args:
            game_state: Final game state
            game_history: History of moves in the game
            duration: Game duration in seconds
            
        Returns:
            GameMetrics object
        """
        # Analyze game history
        white_moves = sum(1 for move in game_history if move['player'] == Player.WHITE)
        black_moves = sum(1 for move in game_history if move['player'] == Player.BLACK)
        
        # Count checkers hit (simplified - would need more detailed tracking)
        white_checkers_hit = 0
        black_checkers_hit = 0
        
        # Count bear-off moves (simplified)
        white_bear_off_moves = 0
        black_bear_off_moves = 0
        
        for move_data in game_history:
            for move in move_data.get('action', []):
                if hasattr(move, 'to_point'):
                    if move.to_point == 26:  # White home
                        white_bear_off_moves += 1
                    elif move.to_point == 27:  # Black home
                        black_bear_off_moves += 1
        
        metrics = GameMetrics(
            winner=game_state.winner,
            game_length=len(game_history),
            white_moves=white_moves,
            black_moves=black_moves,
            white_checkers_hit=white_checkers_hit,
            black_checkers_hit=black_checkers_hit,
            white_bear_off_moves=white_bear_off_moves,
            black_bear_off_moves=black_bear_off_moves,
            game_duration=duration
        )
        
        self.game_metrics.append(metrics)
        return metrics
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all recorded games."""
        if not self.game_metrics:
            return {}
            
        game_lengths = [m.game_length for m in self.game_metrics]
        durations = [m.game_duration for m in self.game_metrics]
        
        white_wins = sum(1 for m in self.game_metrics if m.winner == Player.WHITE)
        black_wins = sum(1 for m in self.game_metrics if m.winner == Player.BLACK)
        total_games = len(self.game_metrics)
        
        return {
            'total_games': total_games,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'white_win_rate': white_wins / total_games if total_games > 0 else 0,
            'black_win_rate': black_wins / total_games if total_games > 0 else 0,
            'avg_game_length': np.mean(game_lengths),
            'std_game_length': np.std(game_lengths),
            'min_game_length': np.min(game_lengths),
            'max_game_length': np.max(game_lengths),
            'avg_game_duration': np.mean(durations),
            'total_duration': np.sum(durations),
            'avg_white_moves': np.mean([m.white_moves for m in self.game_metrics]),
            'avg_black_moves': np.mean([m.black_moves for m in self.game_metrics]),
            'avg_white_bear_offs': np.mean([m.white_bear_off_moves for m in self.game_metrics]),
            'avg_black_bear_offs': np.mean([m.black_bear_off_moves for m in self.game_metrics])
        }
        
    def compare_agents(self, agent1_games: List[int], agent2_games: List[int]) -> Dict[str, Any]:
        """Compare performance between two agents.
        
        Args:
            agent1_games: Indices of games where agent1 played
            agent2_games: Indices of games where agent2 played
            
        Returns:
            Comparison statistics
        """
        if not agent1_games or not agent2_games:
            return {}
            
        agent1_metrics = [self.game_metrics[i] for i in agent1_games if i < len(self.game_metrics)]
        agent2_metrics = [self.game_metrics[i] for i in agent2_games if i < len(self.game_metrics)]
        
        agent1_lengths = [m.game_length for m in agent1_metrics]
        agent2_lengths = [m.game_length for m in agent2_metrics]
        
        return {
            'agent1_avg_length': np.mean(agent1_lengths) if agent1_lengths else 0,
            'agent2_avg_length': np.mean(agent2_lengths) if agent2_lengths else 0,
            'agent1_games': len(agent1_metrics),
            'agent2_games': len(agent2_metrics),
            'length_difference': np.mean(agent1_lengths) - np.mean(agent2_lengths) if agent1_lengths and agent2_lengths else 0
        }
        
    def reset(self):
        """Reset all recorded metrics."""
        self.game_metrics = []
