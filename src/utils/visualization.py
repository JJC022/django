"""Visualization utilities for backgammon analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import pandas as pd


class Visualizer:
    """Visualization utilities for training and analysis."""
    
    @staticmethod
    def plot_training_progress(stats: Dict[str, List], title: str = "Training Progress"):
        """Plot training progress with multiple metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win rates over time
        if 'win_rates' in stats and stats['win_rates']:
            axes[0, 0].plot(stats.get('game_numbers', range(len(stats['win_rates']))), 
                           stats['win_rates'], 'b-', linewidth=2)
            axes[0, 0].set_title('Win Rate Over Time')
            axes[0, 0].set_xlabel('Games')
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
        
        # Learning curve (if available)
        if 'losses' in stats and stats['losses']:
            axes[0, 1].plot(stats.get('game_numbers', range(len(stats['losses']))), 
                           stats['losses'], 'r-', linewidth=2)
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Games')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Game length over time
        if 'avg_game_length' in stats and stats['avg_game_length']:
            axes[1, 0].plot(stats.get('game_numbers', range(len(stats['avg_game_length']))), 
                           stats['avg_game_length'], 'g-', linewidth=2)
            axes[1, 0].set_title('Average Game Length')
            axes[1, 0].set_xlabel('Games')
            axes[1, 0].set_ylabel('Moves per Game')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Exploration rate (if available)
        if 'epsilon' in stats and stats['epsilon']:
            axes[1, 1].plot(stats.get('game_numbers', range(len(stats['epsilon']))), 
                           stats['epsilon'], 'm-', linewidth=2)
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].set_xlabel('Games')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_agent_comparison(agents_stats: Dict[str, Dict], metric: str = 'win_rate'):
        """Compare multiple agents on a specific metric."""
        agent_names = list(agents_stats.keys())
        values = [agents_stats[agent].get(metric, 0) for agent in agent_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(agent_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(agent_names))))
        
        plt.title(f'Agent Comparison: {metric.replace("_", " ").title()}')
        plt.xlabel('Agent')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_position_analysis(position_features: np.ndarray, feature_names: List[str]):
        """Visualize position features."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature values
        ax1.bar(range(len(position_features)), position_features)
        ax1.set_title('Position Features')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Value')
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        
        # Feature importance (normalized)
        normalized_features = position_features / (np.max(np.abs(position_features)) + 1e-8)
        colors = ['red' if x < 0 else 'blue' for x in normalized_features]
        ax2.barh(range(len(normalized_features)), normalized_features, color=colors)
        ax2.set_title('Normalized Feature Values')
        ax2.set_xlabel('Normalized Value')
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_game_tree_analysis(move_evaluations: List[Tuple[str, float]]):
        """Visualize move evaluations for position analysis."""
        moves, values = zip(*move_evaluations)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(moves)), values, 
                      color=['green' if v > 0 else 'red' for v in values])
        
        plt.title('Move Evaluation Analysis')
        plt.xlabel('Move')
        plt.ylabel('Evaluation Score')
        plt.xticks(range(len(moves)), moves, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if value > 0 else -0.03),
                    f'{value:.3f}', ha='center', 
                    va='bottom' if value > 0 else 'top')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def create_training_dashboard(training_data: Dict[str, Any]):
        """Create a comprehensive training dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Win rate progression
        ax1 = fig.add_subplot(gs[0, :2])
        if 'win_rates' in training_data:
            ax1.plot(training_data['game_numbers'], training_data['win_rates'], 'b-', linewidth=2)
            ax1.set_title('Win Rate Progression', fontsize=14)
            ax1.set_xlabel('Games')
            ax1.set_ylabel('Win Rate')
            ax1.grid(True, alpha=0.3)
        
        # Loss progression
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'losses' in training_data:
            ax2.plot(training_data['game_numbers'], training_data['losses'], 'r-', linewidth=2)
            ax2.set_title('Training Loss', fontsize=14)
            ax2.set_xlabel('Games')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        
        # Game statistics
        ax3 = fig.add_subplot(gs[1, :2])
        if 'avg_game_length' in training_data:
            ax3.plot(training_data['game_numbers'], training_data['avg_game_length'], 'g-', linewidth=2)
            ax3.set_title('Average Game Length', fontsize=14)
            ax3.set_xlabel('Games')
            ax3.set_ylabel('Moves per Game')
            ax3.grid(True, alpha=0.3)
        
        # Performance metrics
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'performance_metrics' in training_data:
            metrics = training_data['performance_metrics']
            ax4.bar(metrics.keys(), metrics.values())
            ax4.set_title('Final Performance Metrics', fontsize=14)
            ax4.tick_params(axis='x', rotation=45)
        
        # Training time analysis
        ax5 = fig.add_subplot(gs[2, :])
        if 'training_time' in training_data:
            ax5.plot(training_data['game_numbers'], training_data['training_time'], 'purple', linewidth=2)
            ax5.set_title('Cumulative Training Time', fontsize=14)
            ax5.set_xlabel('Games')
            ax5.set_ylabel('Time (seconds)')
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Training Dashboard', fontsize=18)
        return fig
