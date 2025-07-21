"""Main backgammon environment with pygame visualization."""

import pygame
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .game_state import GameState, Move, Player
from .board import Board


class BackgammonEnv:
    """Backgammon environment with pygame visualization and RL interface."""
    
    # Colors
    BROWN = (139, 69, 19)
    LIGHT_BROWN = (205, 133, 63)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    GRAY = (128, 128, 128)
    
    def __init__(self, width: int = 1200, height: int = 800, render_mode: str = "human"):
        """Initialize the backgammon environment.
        
        Args:
            width: Window width
            height: Window height
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Game state
        self.game_state = GameState()
        self.selected_point = None
        self.highlighted_moves = []
        
        # Pygame setup
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Backgammon RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
        else:
            self.screen = None
            self.clock = None
            self.font = None
            self.small_font = None
            
        # Board layout calculations
        self.board_margin = 50
        self.board_width = width - 2 * self.board_margin
        self.board_height = height - 2 * self.board_margin
        self.point_width = self.board_width // 15  # 12 points + bar + 2 home areas
        self.point_height = self.board_height // 3
        self.checker_radius = min(self.point_width // 3, 20)
        
        # Calculate point positions
        self._calculate_point_positions()
        
    def _calculate_point_positions(self):
        """Calculate the screen positions for each board point."""
        self.point_positions = {}
        
        # Top row (points 12-23)
        for i in range(12):
            point_num = 12 + i
            if i < 6:
                x = self.board_margin + i * self.point_width
            else:
                x = self.board_margin + (i + 1) * self.point_width  # Skip bar
            y = self.board_margin
            self.point_positions[point_num] = (x, y, self.point_width, self.point_height)
            
        # Bottom row (points 0-11)
        for i in range(12):
            point_num = 11 - i
            if i < 6:
                x = self.board_margin + i * self.point_width
            else:
                x = self.board_margin + (i + 1) * self.point_width  # Skip bar
            y = self.board_margin + 2 * self.point_height
            self.point_positions[point_num] = (x, y, self.point_width, self.point_height)
            
        # Bar positions
        bar_x = self.board_margin + 6 * self.point_width
        self.point_positions[24] = (bar_x, self.board_margin, self.point_width, self.point_height)  # White bar
        self.point_positions[25] = (bar_x, self.board_margin + 2 * self.point_height, self.point_width, self.point_height)  # Black bar
        
        # Home positions
        home_x = self.board_margin + 13 * self.point_width
        self.point_positions[26] = (home_x, self.board_margin, 2 * self.point_width, self.point_height)  # White home
        self.point_positions[27] = (home_x, self.board_margin + 2 * self.point_height, 2 * self.point_width, self.point_height)  # Black home
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.game_state.reset()
        self.selected_point = None
        self.highlighted_moves = []
        return self._get_observation()
        
    def step(self, action: List[Move]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute a move sequence and return the new state.
        
        Args:
            action: List of moves to execute
            
        Returns:
            observation: New state observation
            reward: Reward for the action
            done: Whether the game is over
            info: Additional information
        """
        # Execute the moves
        success = self.game_state.make_move_sequence(action)
        
        # Calculate reward
        reward = self._calculate_reward(success)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if game is done
        done = self.game_state.game_over
        
        # Additional info
        info = {
            "legal_move": success,
            "winner": self.game_state.winner.value if self.game_state.winner else None,
            "current_player": self.game_state.current_player.value
        }
        
        return observation, reward, done, info
        
    def _get_observation(self) -> np.ndarray:
        """Get the current state observation."""
        return np.array(self.game_state.get_state_vector(), dtype=np.float32)
        
    def _calculate_reward(self, legal_move: bool) -> float:
        """Calculate reward for the current state."""
        if not legal_move:
            return -1.0  # Penalty for illegal move
            
        if self.game_state.game_over:
            if self.game_state.winner == self.game_state.current_player:
                return 1.0  # Win
            else:
                return -1.0  # Loss
                
        return 0.0  # Neutral for ongoing game
        
    def get_legal_actions(self) -> List[List[Move]]:
        """Get all legal move sequences for the current player."""
        if not self.game_state.dice or self.game_state.dice == (0, 0):
            self.game_state.roll_dice()
            
        dice_values = self.game_state.get_available_dice_values()
        return self.game_state.generate_legal_moves(self.game_state.current_player, dice_values)
        
    def render(self, mode: str = None):
        """Render the current game state."""
        if mode is None:
            mode = self.render_mode
            
        if mode == "human" and self.screen is not None:
            self._render_pygame()
        elif mode == "rgb_array":
            return self._render_rgb_array()
            
    def _render_pygame(self):
        """Render using pygame."""
        self.screen.fill(self.LIGHT_BROWN)
        
        # Draw board background
        board_rect = pygame.Rect(self.board_margin, self.board_margin, 
                               self.board_width, self.board_height)
        pygame.draw.rect(self.screen, self.BROWN, board_rect)
        
        # Draw points
        self._draw_points()
        
        # Draw checkers
        self._draw_checkers()
        
        # Draw UI elements
        self._draw_ui()
        
        pygame.display.flip()
        
    def _draw_points(self):
        """Draw the board points with move highlighting."""
        for point_num in range(24):
            x, y, w, h = self.point_positions[point_num]
            
            # Check if this point is a possible move destination
            is_possible_move = point_num in self.highlighted_moves
            
            # Alternate colors for points
            if is_possible_move:
                color = self.GREEN  # Highlight possible moves in green
            else:
                color = self.LIGHT_BROWN if point_num % 2 == 0 else self.BROWN
            
            # Draw triangular point
            if point_num >= 12:  # Top points
                points = [(x, y + h), (x + w//2, y), (x + w, y + h)]
            else:  # Bottom points
                points = [(x, y), (x + w//2, y + h), (x + w, y)]
                
            pygame.draw.polygon(self.screen, color, points)
            
            # Draw thicker border for possible moves
            border_width = 4 if is_possible_move else 2
            border_color = self.GREEN if is_possible_move else self.BLACK
            pygame.draw.polygon(self.screen, border_color, points, border_width)
            
            # Draw point number
            text_color = self.WHITE if is_possible_move else self.BLACK
            text = self.small_font.render(str(point_num + 1), True, text_color)
            text_rect = text.get_rect(center=(x + w//2, y + h//2))
            self.screen.blit(text, text_rect)
            
    def _draw_checkers(self):
        """Draw checkers on the board with vertical stacking."""
        for point_num in range(28):  # Include bar and home
            checker_count = abs(self.game_state.board.points[point_num])
            if checker_count == 0:
                continue
                
            # Determine checker color
            is_white = self.game_state.board.points[point_num] > 0
            checker_color = self.WHITE if is_white else self.BLACK
            border_color = self.BLACK if is_white else self.WHITE
            
            # Highlight if this point is selected
            is_selected = (self.selected_point == point_num)
            
            # Get position
            if point_num in self.point_positions:
                x, y, w, h = self.point_positions[point_num]
                
                # Calculate checker positions - stack vertically
                checker_spacing = min(self.checker_radius * 2 + 2, h // max(checker_count, 1))
                
                for i in range(checker_count):
                    if point_num < 12:  # Bottom points - stack upward
                        checker_x = x + w // 2
                        checker_y = y + self.checker_radius + i * checker_spacing
                    elif point_num < 24:  # Top points - stack downward
                        checker_x = x + w // 2
                        checker_y = y + h - self.checker_radius - i * checker_spacing
                    elif point_num == 24:  # White bar
                        checker_x = x + w // 2
                        checker_y = y + self.checker_radius + i * checker_spacing
                    elif point_num == 25:  # Black bar
                        checker_x = x + w // 2
                        checker_y = y + h - self.checker_radius - i * checker_spacing
                    else:  # Home areas (26, 27)
                        checker_x = x + w // 2
                        checker_y = y + h // 2 + (i - checker_count // 2) * checker_spacing
                    
                    # Ensure checkers stay within bounds
                    checker_y = max(y + self.checker_radius, min(y + h - self.checker_radius, checker_y))
                    
                    # Draw selection highlight for top checker
                    if is_selected and i == checker_count - 1:
                        pygame.draw.circle(self.screen, self.YELLOW, 
                                         (int(checker_x), int(checker_y)), self.checker_radius + 4, 3)
                    
                    # Draw checker
                    pygame.draw.circle(self.screen, checker_color, 
                                     (int(checker_x), int(checker_y)), self.checker_radius)
                    pygame.draw.circle(self.screen, border_color, 
                                     (int(checker_x), int(checker_y)), self.checker_radius, 2)
                    
                    # Add number if more than 5 checkers
                    if checker_count > 5 and i == min(4, checker_count - 1):
                        text = self.small_font.render(str(checker_count), True, 
                                                     border_color if is_white else checker_color)
                        text_rect = text.get_rect(center=(int(checker_x), int(checker_y)))
                        self.screen.blit(text, text_rect)
                                     
    def _draw_ui(self):
        """Draw UI elements like dice, current player, etc."""
        # Current player
        player_text = f"Current Player: {'White' if self.game_state.current_player == Player.WHITE else 'Black'}"
        text = self.font.render(player_text, True, self.BLACK)
        self.screen.blit(text, (10, 10))
        
        # Dice
        if self.game_state.dice != (0, 0):
            dice_text = f"Dice: {self.game_state.dice[0]}, {self.game_state.dice[1]}"
            text = self.font.render(dice_text, True, self.BLACK)
            self.screen.blit(text, (10, 40))
            
        # Game status
        if self.game_state.game_over:
            winner_text = f"Winner: {'White' if self.game_state.winner == Player.WHITE else 'Black'}"
            text = self.font.render(winner_text, True, self.RED)
            text_rect = text.get_rect(center=(self.width//2, 70))
            self.screen.blit(text, text_rect)
            
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for headless operation."""
        if self.screen is None:
            # Create temporary surface for rendering
            surface = pygame.Surface((self.width, self.height))
            # Render to surface (similar to _render_pygame but to surface)
            # Return as numpy array
            return pygame.surfarray.array3d(surface).transpose((1, 0, 2))
        else:
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
            
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """Handle mouse click for interactive play."""
        clicked_point = self._get_point_from_position(pos)
        
        if clicked_point is not None:
            if self.selected_point is None:
                # Select a point to move from
                if self.game_state.board.can_move_from(clicked_point, self.game_state.current_player):
                    self.selected_point = clicked_point
                    self._highlight_possible_moves(clicked_point)
                    return True
            else:
                # Try to move to clicked point
                if self._try_move(self.selected_point, clicked_point):
                    self.selected_point = None
                    self.highlighted_moves = []
                    return True
                else:
                    # Select new point or deselect
                    if clicked_point == self.selected_point:
                        self.selected_point = None
                        self.highlighted_moves = []
                    elif self.game_state.board.can_move_from(clicked_point, self.game_state.current_player):
                        self.selected_point = clicked_point
                        self._highlight_possible_moves(clicked_point)
                    return True
                    
        return False
        
    def _get_point_from_position(self, pos: Tuple[int, int]) -> Optional[int]:
        """Get board point number from screen position."""
        x, y = pos
        
        for point_num, (px, py, pw, ph) in self.point_positions.items():
            if px <= x <= px + pw and py <= y <= py + ph:
                return point_num
                
        return None
        
    def _highlight_possible_moves(self, from_point: int):
        """Highlight possible moves from selected point."""
        self.highlighted_moves = []
        dice_values = self.game_state.get_available_dice_values()
        
        for die_value in dice_values:
            direction = 1 if self.game_state.current_player == Player.WHITE else -1
            to_point = from_point + die_value * direction
            
            # Check bearing off
            if self.game_state.board.all_checkers_in_home_board(self.game_state.current_player):
                if (self.game_state.current_player == Player.WHITE and from_point >= 18 and to_point >= 24) or \
                   (self.game_state.current_player == Player.BLACK and from_point <= 5 and to_point < 0):
                    to_point = 26 if self.game_state.current_player == Player.WHITE else 27
                    
            if (0 <= to_point <= 23 and self.game_state.board.can_move_to(to_point, self.game_state.current_player)) or \
               to_point in [26, 27]:
                self.highlighted_moves.append(to_point)
                
    def _try_move(self, from_point: int, to_point: int) -> bool:
        """Try to make a move between two points."""
        dice_values = self.game_state.get_available_dice_values()
        direction = 1 if self.game_state.current_player == Player.WHITE else -1
        
        # Calculate required die value
        if to_point in [26, 27]:  # Bearing off
            if self.game_state.current_player == Player.WHITE:
                die_value = max(1, 24 - from_point)
            else:
                die_value = max(1, from_point + 1)
        else:
            die_value = abs(to_point - from_point)
            
        if die_value in dice_values:
            move = Move(from_point, to_point, die_value)
            return self.game_state.make_move_sequence([move])
            
        return False
        
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.quit()
