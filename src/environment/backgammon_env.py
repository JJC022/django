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
        self.remaining_dice = []
        self.current_turn_moves = []
        
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
        self.remaining_dice = []
        self.current_turn_moves = []
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
        
    def start_new_turn(self):
        """Start a new turn by rolling dice and setting up remaining dice."""
        self.game_state.roll_dice()
        self.remaining_dice = self.game_state.get_available_dice_values()
        self.current_turn_moves = []
        self.selected_point = None
        self.highlighted_moves = []
        
    def get_legal_actions(self) -> List[List[Move]]:
        """Get all legal move sequences for the current player."""
        if not self.remaining_dice:
            self.start_new_turn()
            
        return self.game_state.generate_legal_moves(self.game_state.current_player, self.remaining_dice)
        
    def get_legal_single_moves(self) -> List[Move]:
        """Get legal single moves for the current dice state."""
        if not self.remaining_dice:
            return []
            
        legal_moves = []
        for die_value in set(self.remaining_dice):  # Use set to avoid duplicates
            # Try each point as source
            for from_point in range(28):  # Include bar and home
                if not self.game_state.board.can_move_from(from_point, self.game_state.current_player):
                    continue
                    
                # Calculate destination based on player direction
                if self.game_state.current_player == Player.WHITE:
                    # White moves from low to high points (1→24)
                    to_point = from_point + die_value
                else:
                    # Black moves from high to low points (24→1)
                    to_point = from_point - die_value
                
                # Handle bearing off
                if self.game_state.board.all_checkers_in_home_board(self.game_state.current_player):
                    if (self.game_state.current_player == Player.WHITE and from_point >= 18 and to_point >= 24) or \
                       (self.game_state.current_player == Player.BLACK and from_point <= 5 and to_point < 0):
                        to_point = 26 if self.game_state.current_player == Player.WHITE else 27
                        
                # Check if move is legal
                if (0 <= to_point <= 23 and self.game_state.board.can_move_to(to_point, self.game_state.current_player)) or \
                   to_point in [26, 27]:
                    move = Move(from_point, to_point, die_value)
                    legal_moves.append(move)
                    
        return legal_moves
        
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
        # Current player with color indicator
        player_name = 'White' if self.game_state.current_player == Player.WHITE else 'Black'
        player_color = self.WHITE if self.game_state.current_player == Player.WHITE else self.BLACK
        
        # Draw player indicator box
        pygame.draw.rect(self.screen, player_color, (10, 10, 30, 20))
        pygame.draw.rect(self.screen, self.BLACK, (10, 10, 30, 20), 2)
        
        player_text = f"Current Player: {player_name}"
        text = self.font.render(player_text, True, self.BLACK)
        self.screen.blit(text, (50, 10))
        
        # Dice with visual dice representation
        if self.game_state.dice != (0, 0):
            dice_text = f"Dice: {self.game_state.dice[0]}, {self.game_state.dice[1]}"
            text = self.font.render(dice_text, True, self.BLACK)
            self.screen.blit(text, (10, 40))
            
            # Show remaining dice
            if self.remaining_dice:
                remaining_text = f"Remaining: {self.remaining_dice}"
                text = self.small_font.render(remaining_text, True, self.BLUE)
                self.screen.blit(text, (10, 65))
            
            # Draw visual dice (highlight used ones)
            dice_values = self.game_state.get_available_dice_values()
            dice_x_start = 200
            for i, die_value in enumerate(dice_values):
                dice_x = dice_x_start + i * 35
                dice_y = 40
                
                # Check if this die is still available
                is_available = die_value in self.remaining_dice
                self._draw_die(dice_x, dice_y, die_value, available=is_available)
        
        # Selected piece information
        if self.selected_point is not None:
            selected_text = f"Selected: Point {self.selected_point + 1}"
            if self.selected_point == 24:
                selected_text = "Selected: White Bar"
            elif self.selected_point == 25:
                selected_text = "Selected: Black Bar"
            elif self.selected_point >= 26:
                selected_text = "Selected: Home"
                
            text = self.font.render(selected_text, True, self.BLUE)
            self.screen.blit(text, (10, 70))
            
            # Show number of possible moves
            if self.highlighted_moves:
                moves_text = f"Possible moves: {len(self.highlighted_moves)}"
                text = self.small_font.render(moves_text, True, self.BLUE)
                self.screen.blit(text, (10, 95))
        
        # Instructions
        if not self.game_state.game_over:
            if not self.remaining_dice:
                instruction = "Press SPACE to roll dice for new turn"
            elif self.selected_point is None:
                instruction = "Click on a checker to select it"
            else:
                instruction = "Click on destination to move"
            
            text = self.small_font.render(instruction, True, self.GRAY)
            self.screen.blit(text, (10, self.height - 30))
            
            # Show turn status
            if self.remaining_dice:
                turn_status = f"Turn in progress - {len(self.remaining_dice)} dice remaining"
            else:
                turn_status = f"Turn complete - press SPACE for next player"
            
            status_text = self.small_font.render(turn_status, True, self.BLUE)
            self.screen.blit(status_text, (10, self.height - 50))
        
        # Game status
        if self.game_state.game_over:
            winner_text = f"Winner: {'White' if self.game_state.winner == Player.WHITE else 'Black'}"
            text = self.font.render(winner_text, True, self.RED)
            text_rect = text.get_rect(center=(self.width//2, 70))
            
            # Draw background for winner text
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, self.WHITE, bg_rect)
            pygame.draw.rect(self.screen, self.RED, bg_rect, 3)
            self.screen.blit(text, text_rect)
            
        # Highlight bearing off areas if applicable
        if self.selected_point is not None:
            player = self.game_state.current_player
            if self.game_state.board.all_checkers_in_home_board(player):
                home_point = 26 if player == Player.WHITE else 27
                if home_point in self.point_positions:
                    x, y, w, h = self.point_positions[home_point]
                    pygame.draw.rect(self.screen, self.GREEN, (x, y, w, h), 4)
                    
                    # Add "BEAR OFF" text
                    bear_text = self.small_font.render("BEAR OFF", True, self.GREEN)
                    text_rect = bear_text.get_rect(center=(x + w//2, y + h//2))
                    self.screen.blit(bear_text, text_rect)
    
    def _draw_die(self, x: int, y: int, value: int, available: bool = True):
        """Draw a visual representation of a die."""
        die_size = 25
        
        # Draw die background - different colors for available vs used
        die_rect = pygame.Rect(x, y, die_size, die_size)
        if available:
            bg_color = self.WHITE
            border_color = self.BLACK
            dot_color = self.BLACK
        else:
            bg_color = self.GRAY
            border_color = self.BLACK
            dot_color = self.WHITE
            
        pygame.draw.rect(self.screen, bg_color, die_rect)
        pygame.draw.rect(self.screen, border_color, die_rect, 2)
        
        # Draw dots based on value
        dot_radius = 3
        center_x, center_y = x + die_size // 2, y + die_size // 2
        
        # Dot positions relative to center
        positions = {
            1: [(0, 0)],
            2: [(-6, -6), (6, 6)],
            3: [(-6, -6), (0, 0), (6, 6)],
            4: [(-6, -6), (6, -6), (-6, 6), (6, 6)],
            5: [(-6, -6), (6, -6), (0, 0), (-6, 6), (6, 6)],
            6: [(-6, -6), (6, -6), (-6, 0), (6, 0), (-6, 6), (6, 6)]
        }
        
        if value in positions:
            for dx, dy in positions[value]:
                pygame.draw.circle(self.screen, dot_color, 
                                 (center_x + dx, center_y + dy), dot_radius)
            
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
                move_made = self._try_single_move(self.selected_point, clicked_point)
                if move_made:
                    self.selected_point = None
                    self.highlighted_moves = []
                    
                    # Check if turn is complete (no more dice or no legal moves)
                    if not self.remaining_dice or not self.get_legal_single_moves():
                        self._end_turn()
                    
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
        
        # Use remaining dice instead of all dice
        for die_value in set(self.remaining_dice):
            # Calculate destination based on player direction
            if self.game_state.current_player == Player.WHITE:
                # White moves from low to high points (1→24)
                to_point = from_point + die_value
            else:
                # Black moves from high to low points (24→1)
                to_point = from_point - die_value
            
            # Check bearing off
            if self.game_state.board.all_checkers_in_home_board(self.game_state.current_player):
                if (self.game_state.current_player == Player.WHITE and from_point >= 18 and to_point >= 24) or \
                   (self.game_state.current_player == Player.BLACK and from_point <= 5 and to_point < 0):
                    to_point = 26 if self.game_state.current_player == Player.WHITE else 27
                    
            if (0 <= to_point <= 23 and self.game_state.board.can_move_to(to_point, self.game_state.current_player)) or \
               to_point in [26, 27]:
                self.highlighted_moves.append(to_point)
                
    def _try_single_move(self, from_point: int, to_point: int) -> bool:
        """Try to make a single move between two points."""
        # Calculate required die value based on move direction
        if to_point in [26, 27]:  # Bearing off
            if self.game_state.current_player == Player.WHITE:
                # White bearing off from home board (points 18-23)
                die_value = max(1, 24 - from_point)
            else:
                # Black bearing off from home board (points 0-5)
                die_value = max(1, from_point + 1)
        else:
            # Regular move - calculate die value based on direction
            if self.game_state.current_player == Player.WHITE:
                # White moves forward (increasing point numbers)
                die_value = to_point - from_point
            else:
                # Black moves backward (decreasing point numbers)
                die_value = from_point - to_point
            
        # Check if this die value is available
        if die_value in self.remaining_dice:
            # Make the move
            if self.game_state.board.move_checker(from_point, to_point, self.game_state.current_player):
                # Remove the used die from remaining dice
                self.remaining_dice.remove(die_value)
                
                # Record the move
                move = Move(from_point, to_point, die_value)
                self.current_turn_moves.append(move)
                
                # Check for game over
                if self.game_state.board.is_game_over():
                    self.game_state.game_over = True
                    self.game_state.winner = self.game_state.board.get_winner()
                
                return True
            
        return False
        
    def _end_turn(self):
        """End the current turn and switch players."""
        # Record the complete turn in history
        if self.current_turn_moves:
            from .game_state import Turn
            turn = Turn(self.current_turn_moves, self.game_state.dice, self.game_state.current_player)
            self.game_state.turn_history.append(turn)
        
        # Switch players
        if not self.game_state.game_over:
            self.game_state.current_player = (
                Player.BLACK if self.game_state.current_player == Player.WHITE else Player.WHITE
            )
        
        # Reset turn state
        self.remaining_dice = []
        self.current_turn_moves = []
        self.selected_point = None
        self.highlighted_moves = []
        
    def _try_move(self, from_point: int, to_point: int) -> bool:
        """Try to make a move between two points (legacy method for compatibility)."""
        return self._try_single_move(from_point, to_point)
        
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.quit()
