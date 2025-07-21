"""Test script for the backgammon environment."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import BackgammonEnv
from src.environment.board import Player
import pygame


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Backgammon Environment...")
    
    # Create environment
    env = BackgammonEnv(render_mode="human")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial player: {env.game_state.current_player}")
    
    # Roll dice and get legal moves
    env.game_state.roll_dice()
    print(f"Rolled dice: {env.game_state.dice}")
    
    legal_moves = env.get_legal_actions()
    print(f"Number of legal move sequences: {len(legal_moves)}")
    
    if legal_moves:
        print("First few legal moves:")
        for i, move_seq in enumerate(legal_moves[:3]):
            print(f"  Move {i+1}: {[(m.from_point, m.to_point, m.die_value) for m in move_seq]}")
    
    # Test rendering
    env.render()
    
    return env


def interactive_test():
    """Run interactive test with pygame."""
    env = test_basic_functionality()
    
    print("\nStarting interactive test...")
    print("Click on checkers to select them, then click destination to move.")
    print("Press SPACE to roll dice, ESC to quit.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not env.game_state.game_over:
                        # Use the environment's dice management system
                        if not env.remaining_dice:
                            env.start_new_turn()
                            print(f"Rolled: {env.game_state.dice}")
                            print(f"Available dice: {env.remaining_dice}")
                        else:
                            print(f"Still have remaining dice: {env.remaining_dice}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    env.handle_click(event.pos)
        
        env.render()
        env.clock.tick(60)
    
    env.close()


if __name__ == "__main__":
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
