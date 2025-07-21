"""Tournament system for evaluating backgammon agents."""

import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..agents.base_agent import BaseAgent
from ..training.trainer import Trainer


@dataclass
class MatchResult:
    """Result of a match between two agents."""
    white_agent: str
    black_agent: str
    white_wins: int
    black_wins: int
    draws: int
    total_games: int
    white_win_rate: float
    black_win_rate: float
    avg_game_length: float
    total_time: float


class Tournament:
    """Tournament system for comprehensive agent evaluation."""
    
    def __init__(self, trainer: Trainer):
        """Initialize tournament system.
        
        Args:
            trainer: Trainer instance for running games
        """
        self.trainer = trainer
        self.results = []
        
    def round_robin(self, agents: List[BaseAgent], games_per_match: int = 100) -> List[MatchResult]:
        """Run a round-robin tournament between all agents.
        
        Args:
            agents: List of agents to compete
            games_per_match: Number of games per matchup
            
        Returns:
            List of match results
        """
        print(f"Starting round-robin tournament with {len(agents)} agents...")
        print(f"Games per match: {games_per_match}")
        
        results = []
        total_matches = len(agents) * (len(agents) - 1)
        
        with tqdm(total=total_matches, desc="Tournament Progress") as pbar:
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents):
                    if i != j:  # Don't play against self
                        result = self._play_match(agent1, agent2, games_per_match)
                        results.append(result)
                        pbar.update(1)
                        
        self.results = results
        return results
        
    def _play_match(self, white_agent: BaseAgent, black_agent: BaseAgent, 
                   num_games: int) -> MatchResult:
        """Play a match between two agents."""
        # Reset agent statistics
        white_agent.reset_stats()
        black_agent.reset_stats()
        
        start_time = time.time()
        game_lengths = []
        
        for _ in range(num_games):
            game_result = self.trainer.play_game(white_agent, black_agent)
            game_lengths.append(game_result['moves_count'])
            
        match_time = time.time() - start_time
        
        # Calculate draws (games that didn't end in victory)
        draws = num_games - white_agent.wins - black_agent.wins
        
        return MatchResult(
            white_agent=white_agent.name,
            black_agent=black_agent.name,
            white_wins=white_agent.wins,
            black_wins=black_agent.wins,
            draws=draws,
            total_games=num_games,
            white_win_rate=white_agent.get_win_rate(),
            black_win_rate=black_agent.get_win_rate(),
            avg_game_length=sum(game_lengths) / len(game_lengths) if game_lengths else 0,
            total_time=match_time
        )
        
    def get_leaderboard(self) -> pd.DataFrame:
        """Generate leaderboard from tournament results."""
        if not self.results:
            return pd.DataFrame()
            
        # Collect all agent names
        agents = set()
        for result in self.results:
            agents.add(result.white_agent)
            agents.add(result.black_agent)
            
        # Calculate statistics for each agent
        leaderboard_data = []
        
        for agent in agents:
            total_games = 0
            total_wins = 0
            total_losses = 0
            total_draws = 0
            
            for result in self.results:
                if result.white_agent == agent:
                    total_games += result.total_games
                    total_wins += result.white_wins
                    total_losses += result.black_wins
                    total_draws += result.draws
                elif result.black_agent == agent:
                    total_games += result.total_games
                    total_wins += result.black_wins
                    total_losses += result.white_wins
                    total_draws += result.draws
                    
            win_rate = total_wins / total_games if total_games > 0 else 0
            
            leaderboard_data.append({
                'Agent': agent,
                'Games': total_games,
                'Wins': total_wins,
                'Losses': total_losses,
                'Draws': total_draws,
                'Win Rate': win_rate,
                'Score': total_wins - total_losses  # Simple scoring system
            })
            
        df = pd.DataFrame(leaderboard_data)
        return df.sort_values('Win Rate', ascending=False)
        
    def plot_results(self, save_path: str = None):
        """Create visualizations of tournament results."""
        if not self.results:
            print("No results to plot")
            return
            
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Win rate heatmap
        agents = list(set([r.white_agent for r in self.results] + [r.black_agent for r in self.results]))
        win_matrix = pd.DataFrame(index=agents, columns=agents, dtype=float)
        
        for result in self.results:
            win_matrix.loc[result.white_agent, result.black_agent] = result.white_win_rate
            win_matrix.loc[result.black_agent, result.white_agent] = result.black_win_rate
            
        # Fill diagonal with 0.5 (self-play would be 50%)
        for agent in agents:
            win_matrix.loc[agent, agent] = 0.5
            
        sns.heatmap(win_matrix.astype(float), annot=True, fmt='.2f', 
                   cmap='RdYlBu_r', center=0.5, ax=axes[0, 0])
        axes[0, 0].set_title('Win Rate Matrix (Row vs Column)')
        
        # 2. Overall win rates
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            axes[0, 1].bar(leaderboard['Agent'], leaderboard['Win Rate'])
            axes[0, 1].set_title('Overall Win Rates')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
        # 3. Game length distribution
        game_lengths = [r.avg_game_length for r in self.results]
        axes[1, 0].hist(game_lengths, bins=20, alpha=0.7)
        axes[1, 0].set_title('Distribution of Average Game Lengths')
        axes[1, 0].set_xlabel('Average Moves per Game')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Performance comparison
        if not leaderboard.empty:
            axes[1, 1].scatter(leaderboard['Wins'], leaderboard['Losses'], 
                             s=leaderboard['Games']*2, alpha=0.6)
            for i, agent in enumerate(leaderboard['Agent']):
                axes[1, 1].annotate(agent, 
                                  (leaderboard.iloc[i]['Wins'], leaderboard.iloc[i]['Losses']))
            axes[1, 1].set_xlabel('Total Wins')
            axes[1, 1].set_ylabel('Total Losses')
            axes[1, 1].set_title('Wins vs Losses (bubble size = total games)')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tournament results saved to {save_path}")
        else:
            plt.show()
            
    def save_results(self, filepath: str):
        """Save tournament results to JSON file."""
        results_data = []
        for result in self.results:
            results_data.append({
                'white_agent': result.white_agent,
                'black_agent': result.black_agent,
                'white_wins': result.white_wins,
                'black_wins': result.black_wins,
                'draws': result.draws,
                'total_games': result.total_games,
                'white_win_rate': result.white_win_rate,
                'black_win_rate': result.black_win_rate,
                'avg_game_length': result.avg_game_length,
                'total_time': result.total_time
            })
            
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Tournament results saved to {filepath}")
        
    def load_results(self, filepath: str):
        """Load tournament results from JSON file."""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
            
        self.results = []
        for data in results_data:
            result = MatchResult(**data)
            self.results.append(result)
            
        print(f"Loaded {len(self.results)} tournament results from {filepath}")
        
    def print_summary(self):
        """Print a summary of tournament results."""
        if not self.results:
            print("No tournament results available")
            return
            
        print("\n" + "="*60)
        print("TOURNAMENT SUMMARY")
        print("="*60)
        
        leaderboard = self.get_leaderboard()
        print("\nLEADERBOARD:")
        print(leaderboard.to_string(index=False, float_format='%.3f'))
        
        print(f"\nTOTAL MATCHES: {len(self.results)}")
        total_games = sum(r.total_games for r in self.results)
        print(f"TOTAL GAMES: {total_games}")
        
        avg_game_length = sum(r.avg_game_length * r.total_games for r in self.results) / total_games
        print(f"AVERAGE GAME LENGTH: {avg_game_length:.1f} moves")
        
        total_time = sum(r.total_time for r in self.results)
        print(f"TOTAL TOURNAMENT TIME: {total_time:.1f} seconds")
        
        print("="*60)
