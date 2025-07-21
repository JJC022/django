"""Self-play training for backgammon agents."""

import copy
import random
from typing import List, Dict, Any
from tqdm import tqdm
from .trainer import Trainer
from ..agents.base_agent import BaseAgent
from ..environment import Player


class SelfPlayTrainer(Trainer):
    """Trainer specialized for self-play training."""
    
    def __init__(self, env, save_interval: int = 1000):
        """Initialize self-play trainer.
        
        Args:
            env: Backgammon environment
            save_interval: How often to save agent checkpoints
        """
        super().__init__(env)
        self.save_interval = save_interval
        self.generation = 0
        
    def self_play_training(self, agent: BaseAgent, num_games: int,
                          opponent_pool_size: int = 5,
                          update_pool_interval: int = 200) -> Dict[str, List]:
        """Train agent using self-play against previous versions.
        
        Args:
            agent: Agent to train
            num_games: Total number of training games
            opponent_pool_size: Size of opponent pool (previous versions)
            update_pool_interval: How often to add current agent to pool
            
        Returns:
            Training statistics
        """
        print(f"Starting self-play training for {agent.name}...")
        
        # Initialize opponent pool with copies of the agent
        opponent_pool = []
        for i in range(min(3, opponent_pool_size)):  # Start with fewer opponents
            opponent = copy.deepcopy(agent)
            opponent.name = f"{agent.name}_gen_{i}"
            opponent.set_training(False)  # Opponents don't learn
            opponent_pool.append(opponent)
            
        stats = {
            'game_numbers': [],
            'win_rates': [],
            'opponent_strengths': [],
            'generations': []
        }
        
        games_played = 0
        
        while games_played < num_games:
            # Select random opponent from pool
            opponent = random.choice(opponent_pool)
            
            # Randomly assign colors
            if random.random() < 0.5:
                white_agent, black_agent = agent, opponent
            else:
                white_agent, black_agent = opponent, agent
                
            # Play game
            result = self.play_game(white_agent, black_agent)
            games_played += 1
            
            # Update opponent pool periodically
            if games_played % update_pool_interval == 0:
                self._update_opponent_pool(agent, opponent_pool, opponent_pool_size)
                
            # Log progress
            if games_played % 100 == 0:
                stats['game_numbers'].append(games_played)
                stats['win_rates'].append(agent.get_win_rate())
                stats['generations'].append(self.generation)
                
                print(f"Games: {games_played}/{num_games}, "
                      f"Win rate: {agent.get_win_rate():.3f}, "
                      f"Generation: {self.generation}")
                      
            # Save checkpoint
            if games_played % self.save_interval == 0:
                checkpoint_path = f"models/{agent.name}_checkpoint_{games_played}.pth"
                agent.save(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
        return stats
        
    def _update_opponent_pool(self, current_agent: BaseAgent, 
                            opponent_pool: List[BaseAgent], 
                            max_pool_size: int):
        """Update the opponent pool with current agent version."""
        # Create a copy of current agent for the pool
        new_opponent = copy.deepcopy(current_agent)
        new_opponent.name = f"{current_agent.name}_gen_{self.generation}"
        new_opponent.set_training(False)
        new_opponent.reset_stats()
        
        # Add to pool
        opponent_pool.append(new_opponent)
        
        # Remove oldest if pool is too large
        if len(opponent_pool) > max_pool_size:
            opponent_pool.pop(0)
            
        self.generation += 1
        print(f"Updated opponent pool (generation {self.generation})")
        
    def curriculum_training(self, agent: BaseAgent, curriculum: List[Dict[str, Any]]) -> Dict[str, List]:
        """Train agent using a curriculum of increasingly difficult opponents.
        
        Args:
            agent: Agent to train
            curriculum: List of training phases with different opponents/settings
            
        Returns:
            Training statistics
        """
        print(f"Starting curriculum training for {agent.name}...")
        
        all_stats = {
            'phases': [],
            'cumulative_games': [],
            'win_rates': [],
            'phase_names': []
        }
        
        total_games = 0
        
        for phase_idx, phase in enumerate(curriculum):
            print(f"\n--- Phase {phase_idx + 1}: {phase['name']} ---")
            
            # Extract phase parameters
            opponent_type = phase['opponent']
            num_games = phase['games']
            phase_name = phase['name']
            
            # Create opponent
            if opponent_type == 'self':
                # Self-play
                stats = self.self_play_training(agent, num_games)
            elif opponent_type == 'random':
                # Train against random agent
                from ..agents.random_agent import RandomAgent
                opponent = RandomAgent(Player.BLACK, "RandomOpponent")
                stats = self.train_agents(agent, opponent, num_games)
            else:
                # Custom opponent (assume it's provided)
                stats = self.train_agents(agent, opponent_type, num_games)
                
            total_games += num_games
            
            # Store phase results
            all_stats['phases'].append(phase_idx + 1)
            all_stats['cumulative_games'].append(total_games)
            all_stats['win_rates'].append(agent.get_win_rate())
            all_stats['phase_names'].append(phase_name)
            
            print(f"Phase {phase_idx + 1} completed. Agent win rate: {agent.get_win_rate():.3f}")
            
        return all_stats
