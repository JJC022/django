"""Tournament script for evaluating multiple backgammon agents."""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import BackgammonEnv, Player
from src.agents import TDAgent, RandomAgent
from src.training import Trainer
from src.evaluation import Tournament


def load_config(config_path: str) -> dict:
    """Load tournament configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent_from_config(agent_config: dict, player: Player) -> object:
    """Create agent from configuration."""
    agent_type = agent_config['type']
    
    if agent_type == 'TDAgent':
        agent = TDAgent(
            player=player,
            name=agent_config['name'],
            learning_rate=agent_config.get('learning_rate', 0.001),
            epsilon=0.0,  # No exploration in tournament
            lambda_param=agent_config.get('lambda_param', 0.9)
        )
        
        # Load model if specified
        if 'model_path' in agent_config:
            print(f"Loading model for {agent.name}: {agent_config['model_path']}")
            agent.load(agent_config['model_path'])
            
        agent.set_training(False)
        return agent
        
    elif agent_type == 'RandomAgent':
        return RandomAgent(player=player, name=agent_config['name'])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Main tournament function."""
    parser = argparse.ArgumentParser(description='Run backgammon agent tournament')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to tournament configuration file')
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded tournament configuration from {args.config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = BackgammonEnv(render_mode=None)  # No rendering for tournament
    trainer = Trainer(env)
    tournament = Tournament(trainer)
    
    # Create agents
    agents = []
    for agent_config in config['agents']:
        # Create agent for both colors to test symmetry
        white_agent = create_agent_from_config(agent_config, Player.WHITE)
        black_agent = create_agent_from_config(agent_config, Player.BLACK)
        
        # Rename to distinguish colors
        white_agent.name = f"{agent_config['name']}_White"
        black_agent.name = f"{agent_config['name']}_Black"
        
        agents.extend([white_agent, black_agent])
    
    print(f"Created {len(agents)} agents for tournament:")
    for agent in agents:
        print(f"  - {agent.name}")
    
    # Run tournament
    games_per_match = config['tournament']['games_per_match']
    print(f"\nStarting round-robin tournament with {games_per_match} games per match...")
    
    results = tournament.round_robin(agents, games_per_match)
    
    # Print summary
    tournament.print_summary()
    
    # Save results
    results_file = os.path.join(args.output_dir, 'tournament_results.json')
    tournament.save_results(results_file)
    
    # Create visualizations
    plot_file = os.path.join(args.output_dir, 'tournament_plots.png')
    tournament.plot_results(save_path=plot_file)
    
    # Save leaderboard
    leaderboard = tournament.get_leaderboard()
    leaderboard_file = os.path.join(args.output_dir, 'leaderboard.csv')
    leaderboard.to_csv(leaderboard_file, index=False)
    print(f"Leaderboard saved to {leaderboard_file}")
    
    print(f"\nTournament completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
