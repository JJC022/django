"""Main training script for backgammon RL agents."""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import BackgammonEnv, Player
from src.agents import TDAgent, RandomAgent
from src.training import Trainer, SelfPlayTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(config: dict, player: Player) -> object:
    """Create agent based on configuration."""
    agent_config = config['agent']
    agent_type = agent_config['type']
    
    if agent_type == 'TDAgent':
        return TDAgent(
            player=player,
            name=agent_config['name'],
            learning_rate=agent_config['learning_rate'],
            epsilon=agent_config['epsilon'],
            lambda_param=agent_config['lambda_param']
        )
    elif agent_type == 'RandomAgent':
        return RandomAgent(player=player, name=agent_config['name'])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train backgammon RL agent')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--render', action='store_true',
                       help='Render training games')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Create directories
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    if 'plots_dir' in config['paths']:
        os.makedirs(config['paths']['plots_dir'], exist_ok=True)
    
    # Create environment
    env_config = config['environment']
    render_mode = "human" if args.render else env_config.get('render_mode')
    env = BackgammonEnv(
        width=env_config['width'],
        height=env_config['height'],
        render_mode=render_mode
    )
    
    # Create agent
    agent = create_agent(config, Player.WHITE)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
    
    # Create trainer
    training_config = config['training']
    if training_config['type'] == 'self_play':
        trainer = SelfPlayTrainer(env, save_interval=training_config['save_interval'])
        
        # Start self-play training
        stats = trainer.self_play_training(
            agent=agent,
            num_games=training_config['num_games'],
            opponent_pool_size=training_config['opponent_pool_size'],
            update_pool_interval=training_config['update_pool_interval']
        )
    else:
        # Regular training against specific opponent
        trainer = Trainer(env)
        opponent = RandomAgent(Player.BLACK, "RandomOpponent")
        
        stats = trainer.train_agents(
            white_agent=agent,
            black_agent=opponent,
            num_games=training_config['num_games'],
            eval_interval=training_config['eval_interval']
        )
    
    # Save final model
    final_model_path = os.path.join(config['paths']['models_dir'], f"{agent.name}_final.pth")
    agent.save(final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    # Plot training statistics
    if config['logging']['save_plots']:
        plot_path = os.path.join(config['paths'].get('plots_dir', 'plots'), 
                                f"{agent.name}_training.png")
        trainer.plot_training_stats(stats, save_path=plot_path)
        print(f"Saved training plot: {plot_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
