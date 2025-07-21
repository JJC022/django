"""Training utilities for backgammon agents."""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..environment import BackgammonEnv, Player
from ..agents.base_agent import BaseAgent


class Trainer:
    """Trainer for backgammon RL agents."""
    
    def __init__(self, env: BackgammonEnv):
        """Initialize the trainer.
        
        Args:
            env: Backgammon environment
        """
        self.env = env
        self.training_history = []
        
    def play_game(self, white_agent: BaseAgent, black_agent: BaseAgent, 
                  render: bool = False) -> Dict[str, Any]:
        """Play a single game between two agents.
        
        Args:
            white_agent: Agent playing white
            black_agent: Agent playing black
            render: Whether to render the game
            
        Returns:
            Game result dictionary
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        moves_count = 0
        game_history = []
        
        while not done and moves_count < 1000:  # Prevent infinite games
            current_agent = white_agent if self.env.game_state.current_player == Player.WHITE else black_agent
            
            # Roll dice and get legal moves
            self.env.game_state.roll_dice()
            legal_moves = self.env.get_legal_actions()
            
            if not legal_moves:
                # No legal moves, switch players
                self.env.game_state.current_player = (
                    Player.BLACK if self.env.game_state.current_player == Player.WHITE else Player.WHITE
                )
                continue
                
            # Agent chooses action
            action = current_agent.choose_action(self.env.game_state, legal_moves)
            
            # Store state for learning
            prev_state = self.env.game_state.copy()
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            
            # Update agent
            current_agent.update(prev_state, action, reward, self.env.game_state, done)
            
            # Store move in history
            game_history.append({
                'player': self.env.game_state.current_player,
                'dice': self.env.game_state.dice,
                'action': action,
                'reward': reward
            })
            
            moves_count += 1
            
            if render:
                self.env.render()
                time.sleep(0.1)
                
        # Game over - notify agents
        winner = self.env.game_state.winner
        white_agent.game_over(winner)
        black_agent.game_over(winner)
        
        return {
            'winner': winner,
            'moves_count': moves_count,
            'white_agent': white_agent.name,
            'black_agent': black_agent.name,
            'history': game_history
        }
    
    def train_agents(self, white_agent: BaseAgent, black_agent: BaseAgent,
                    num_games: int, eval_interval: int = 100,
                    render_eval: bool = False) -> Dict[str, List]:
        """Train agents by playing games against each other.
        
        Args:
            white_agent: Agent playing white
            black_agent: Agent playing black
            num_games: Number of training games
            eval_interval: How often to evaluate and log progress
            render_eval: Whether to render evaluation games
            
        Returns:
            Training statistics
        """
        print(f"Training {white_agent.name} vs {black_agent.name} for {num_games} games...")
        
        stats = {
            'game_numbers': [],
            'white_win_rates': [],
            'black_win_rates': [],
            'avg_game_length': [],
            'training_time': []
        }
        
        start_time = time.time()
        
        for game_num in tqdm(range(num_games), desc="Training"):
            # Play game
            result = self.play_game(white_agent, black_agent, render=False)
            
            # Log progress
            if (game_num + 1) % eval_interval == 0:
                elapsed_time = time.time() - start_time
                
                stats['game_numbers'].append(game_num + 1)
                stats['white_win_rates'].append(white_agent.get_win_rate())
                stats['black_win_rates'].append(black_agent.get_win_rate())
                stats['training_time'].append(elapsed_time)
                
                print(f"\nGame {game_num + 1}/{num_games}")
                print(f"  {white_agent}")
                print(f"  {black_agent}")
                print(f"  Time elapsed: {elapsed_time:.1f}s")
                
                # Optional evaluation game with rendering
                if render_eval and game_num > 0:
                    print("  Playing evaluation game...")
                    eval_result = self.play_game(white_agent, black_agent, render=True)
                    print(f"  Evaluation winner: {eval_result['winner']}")
                    
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Final results:")
        print(f"  {white_agent}")
        print(f"  {black_agent}")
        
        return stats
    
    def evaluate_agents(self, agents: List[BaseAgent], num_games: int = 100) -> Dict[str, Any]:
        """Evaluate multiple agents in a round-robin tournament.
        
        Args:
            agents: List of agents to evaluate
            num_games: Number of games per matchup
            
        Returns:
            Tournament results
        """
        print(f"Evaluating {len(agents)} agents in round-robin tournament...")
        
        results = {}
        matchups = []
        
        # Generate all matchups
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    matchups.append((agent1, agent2))
                    
        # Play all matchups
        for white_agent, black_agent in tqdm(matchups, desc="Tournament"):
            matchup_key = f"{white_agent.name}_vs_{black_agent.name}"
            
            # Reset agent stats for this matchup
            white_agent.reset_stats()
            black_agent.reset_stats()
            
            # Play games
            for _ in range(num_games):
                self.play_game(white_agent, black_agent)
                
            # Store results
            results[matchup_key] = {
                'white_agent': white_agent.name,
                'black_agent': black_agent.name,
                'white_wins': white_agent.wins,
                'black_wins': black_agent.wins,
                'white_win_rate': white_agent.get_win_rate(),
                'black_win_rate': black_agent.get_win_rate(),
                'games_played': num_games
            }
            
        return results
    
    def plot_training_stats(self, stats: Dict[str, List], save_path: Optional[str] = None):
        """Plot training statistics.
        
        Args:
            stats: Training statistics from train_agents
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Win rates
        axes[0, 0].plot(stats['game_numbers'], stats['white_win_rates'], 'b-', label='White')
        axes[0, 0].plot(stats['game_numbers'], stats['black_win_rates'], 'r-', label='Black')
        axes[0, 0].set_xlabel('Games')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_title('Win Rates Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training time
        if len(stats['training_time']) > 1:
            axes[0, 1].plot(stats['game_numbers'], stats['training_time'], 'g-')
            axes[0, 1].set_xlabel('Games')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].set_title('Training Time')
            axes[0, 1].grid(True)
        
        # Game length (if available)
        if 'avg_game_length' in stats and stats['avg_game_length']:
            axes[1, 0].plot(stats['game_numbers'], stats['avg_game_length'], 'm-')
            axes[1, 0].set_xlabel('Games')
            axes[1, 0].set_ylabel('Average Moves')
            axes[1, 0].set_title('Average Game Length')
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
