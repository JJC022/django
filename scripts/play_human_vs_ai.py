"""Interactive script for human vs AI gameplay."""

import sys
import os
import pygame
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import BackgammonEnv, Player
from src.agents import TDAgent, RandomAgent


class HumanPlayer:
    """Human player interface for interactive gameplay."""
    
    def __init__(self, player: Player, name: str = "Human"):
        """Initialize human player."""
        self.player = player
        self.name = name
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
    def choose_action(self, game_state, legal_moves):
        """Human chooses action through GUI interaction."""
        # This will be handled by the game loop with mouse clicks
        return None
        
    def update(self, game_state, action, reward, next_state, done):
        """Human doesn't need updates."""
        pass
        
    def game_over(self, winner):
        """Track game results."""
        self.games_played += 1
        if winner == self.player:
            self.wins += 1
        elif winner is not None:
            self.losses += 1
            
    def get_win_rate(self):
        """Get win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


def main():
    """Main function for human vs AI gameplay."""
    parser = argparse.ArgumentParser(description='Play backgammon against AI')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained AI model')
    parser.add_argument('--ai-type', type=str, default='random',
                       choices=['random', 'td'],
                       help='Type of AI opponent')
    parser.add_argument('--human-color', type=str, default='white',
                       choices=['white', 'black'],
                       help='Color for human player')
    
    args = parser.parse_args()
    
    # Create environment
    env = BackgammonEnv(render_mode="human")
    
    # Create players
    human_player = Player.WHITE if args.human_color == 'white' else Player.BLACK
    ai_player = Player.BLACK if args.human_color == 'white' else Player.WHITE
    
    human = HumanPlayer(human_player, "Human")
    
    # Create AI agent
    if args.ai_type == 'td':
        ai = TDAgent(ai_player, "AI-TD")
        if args.model:
            print(f"Loading AI model from {args.model}")
            ai.load(args.model)
        ai.set_training(False)  # No learning during play
        ai.set_epsilon(0.0)     # No exploration
    else:
        ai = RandomAgent(ai_player, "AI-Random")
    
    print(f"Starting game: {human.name} ({human_player.name}) vs {ai.name} ({ai_player.name})")
    print("\nControls:")
    print("- Click on a checker to select it")
    print("- Click on destination to move")
    print("- Press SPACE to roll dice")
    print("- Press R to restart game")
    print("- Press ESC to quit")
    
    # Game loop
    running = True
    game_active = True
    current_move_sequence = []
    dice_rolled = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Restart game
                    env.reset()
                    game_active = True
                    current_move_sequence = []
                    dice_rolled = False
                    print("Game restarted!")
                elif event.key == pygame.K_SPACE and game_active:
                    # Roll dice
                    if not dice_rolled and not env.game_state.game_over:
                        env.game_state.roll_dice()
                        dice_rolled = True
                        print(f"Rolled: {env.game_state.dice}")
                        
                        # If it's AI's turn, let AI play
                        if env.game_state.current_player == ai_player:
                            legal_moves = env.get_legal_actions()
                            if legal_moves:
                                ai_action = ai.choose_action(env.game_state, legal_moves)
                                if ai_action:
                                    env.step(ai_action)
                                    print(f"AI played: {[(m.from_point, m.to_point) for m in ai_action]}")
                                    dice_rolled = False
                                    
                                    if env.game_state.game_over:
                                        winner = env.game_state.winner
                                        print(f"Game Over! Winner: {winner.name if winner else 'Draw'}")
                                        human.game_over(winner)
                                        ai.game_over(winner)
                                        game_active = False
                                        
            elif event.type == pygame.MOUSEBUTTONDOWN and game_active:
                if event.button == 1:  # Left click
                    if env.game_state.current_player == human_player and dice_rolled:
                        # Handle human move
                        if env.handle_click(event.pos):
                            # Check if a complete move was made
                            if env.selected_point is None:  # Move completed
                                dice_rolled = False
                                
                                if env.game_state.game_over:
                                    winner = env.game_state.winner
                                    print(f"Game Over! Winner: {winner.name if winner else 'Draw'}")
                                    human.game_over(winner)
                                    ai.game_over(winner)
                                    game_active = False
        
        # Render the game
        env.render()
        env.clock.tick(60)
        
        # Display game status
        if not game_active and env.game_state.game_over:
            # Show final statistics
            winner = env.game_state.winner
            if winner:
                winner_name = "Human" if winner == human_player else "AI"
                print(f"\nFinal Result: {winner_name} wins!")
            else:
                print("\nGame ended in a draw!")
                
            print(f"Human record: {human.wins}/{human.games_played} ({human.get_win_rate():.1%})")
            print(f"AI record: {ai.wins}/{ai.games_played} ({ai.get_win_rate():.1%})")
    
    env.close()
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
