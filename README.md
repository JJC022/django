# Reinforcement Learning Backgammon

A comprehensive reinforcement learning project implementing various RL algorithms to play backgammon. This project demonstrates advanced RL concepts including self-play, neural network function approximation, and multi-agent training.

## Features

- **Custom Backgammon Environment**: Built with pygame for visualization and interaction
- **Multiple RL Algorithms**: Implementation of various RL approaches (TD-Gammon style, PPO, DQN, etc.)
- **Self-Play Training**: Agents learn by playing against themselves
- **Human vs AI Interface**: Interactive gameplay through pygame GUI
- **Comprehensive Evaluation**: Tournament-style evaluation between different agents
- **Modular Architecture**: Clean separation of environment, agents, and training logic

## Project Structure

```
├── src/
│   ├── environment/          # Backgammon game environment
│   ├── agents/              # RL agent implementations
│   ├── training/            # Training loops and utilities
│   ├── evaluation/          # Evaluation and tournament systems
│   └── utils/               # Shared utilities
├── configs/                 # Configuration files for different experiments
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs and tensorboard data
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Unit tests
└── scripts/                 # Training and evaluation scripts
```

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train an Agent**:
   ```bash
   python scripts/train_agent.py --config configs/td_gammon.yaml
   ```

3. **Play Against AI**:
   ```bash
   python scripts/play_human_vs_ai.py --model models/best_agent.pth
   ```

4. **Run Tournament**:
   ```bash
   python scripts/tournament.py --config configs/tournament.yaml
   ```

## Algorithms Implemented

- **TD-Gammon**: Classic temporal difference learning approach
- **Deep Q-Network (DQN)**: Value-based deep RL
- **Proximal Policy Optimization (PPO)**: Policy gradient method
- **AlphaZero-style**: MCTS + Neural Networks

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Matplotlib
- TensorBoard
- PyYAML

## License

MIT License
