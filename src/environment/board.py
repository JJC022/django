"""Backgammon board representation and logic."""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class Player(Enum):
    """Player enumeration."""
    WHITE = 1
    BLACK = -1


class Board:
    """Backgammon board representation."""
    
    def __init__(self):
        """Initialize the backgammon board with starting position."""
        # Board positions: 0-23 are points, 24=white bar, 25=black bar
        # 26=white home, 27=black home
        self.points = np.zeros(28, dtype=np.int8)
        self.setup_initial_position()
        
    def setup_initial_position(self):
        """Set up the standard backgammon starting position."""
        # White checkers (positive values)
        self.points[0] = 2   # Point 1: 2 white checkers
        self.points[11] = 5  # Point 12: 5 white checkers  
        self.points[16] = 3  # Point 17: 3 white checkers
        self.points[18] = 5  # Point 19: 5 white checkers
        
        # Black checkers (negative values)
        self.points[23] = -2  # Point 24: 2 black checkers
        self.points[12] = -5  # Point 13: 5 black checkers
        self.points[7] = -3   # Point 8: 3 black checkers
        self.points[5] = -5   # Point 6: 5 black checkers
        
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.points = self.points.copy()
        return new_board
        
    def get_checker_count(self, point: int, player: Player) -> int:
        """Get number of checkers for a player at a point."""
        if player == Player.WHITE:
            return max(0, self.points[point])
        else:
            return max(0, -self.points[point])
            
    def can_move_from(self, point: int, player: Player) -> bool:
        """Check if player can move a checker from this point."""
        if player == Player.WHITE:
            return self.points[point] > 0
        else:
            return self.points[point] < 0
            
    def can_move_to(self, point: int, player: Player) -> bool:
        """Check if player can move a checker to this point."""
        if point < 0 or point > 23:
            return False
            
        opponent_checkers = self.get_checker_count(point, 
            Player.BLACK if player == Player.WHITE else Player.WHITE)
        return opponent_checkers <= 1
        
    def move_checker(self, from_point: int, to_point: int, player: Player) -> bool:
        """Move a checker from one point to another."""
        if not self.can_move_from(from_point, player):
            return False
            
        # Handle bearing off (moving to home)
        if to_point == 26 and player == Player.WHITE:
            self.points[from_point] -= 1
            self.points[26] += 1
            return True
        elif to_point == 27 and player == Player.BLACK:
            self.points[from_point] += 1
            self.points[27] -= 1
            return True
            
        if not self.can_move_to(to_point, player):
            return False
            
        # Check for hitting opponent's blot
        opponent = Player.BLACK if player == Player.WHITE else Player.WHITE
        if self.get_checker_count(to_point, opponent) == 1:
            self.hit_checker(to_point, opponent)
            
        # Move the checker
        if player == Player.WHITE:
            self.points[from_point] -= 1
            self.points[to_point] += 1
        else:
            self.points[from_point] += 1
            self.points[to_point] -= 1
            
        return True
        
    def hit_checker(self, point: int, player: Player):
        """Move a checker to the bar when hit."""
        if player == Player.WHITE:
            self.points[point] -= 1
            self.points[24] += 1  # White bar
        else:
            self.points[point] += 1
            self.points[25] -= 1  # Black bar
            
    def checkers_on_bar(self, player: Player) -> int:
        """Get number of checkers on the bar for a player."""
        if player == Player.WHITE:
            return self.points[24]
        else:
            return -self.points[25]
            
    def all_checkers_in_home_board(self, player: Player) -> bool:
        """Check if all checkers are in home board (for bearing off)."""
        if player == Player.WHITE:
            # White home board is points 18-23
            for i in range(18):
                if self.points[i] > 0:
                    return False
            return self.points[24] == 0  # No checkers on bar
        else:
            # Black home board is points 0-5
            for i in range(6, 24):
                if self.points[i] < 0:
                    return False
            return self.points[25] == 0  # No checkers on bar
            
    def checkers_borne_off(self, player: Player) -> int:
        """Get number of checkers borne off."""
        if player == Player.WHITE:
            return self.points[26]
        else:
            return -self.points[27]
            
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.checkers_borne_off(Player.WHITE) == 15 or \
               self.checkers_borne_off(Player.BLACK) == 15
               
    def get_winner(self) -> Optional[Player]:
        """Get the winner if game is over."""
        if self.checkers_borne_off(Player.WHITE) == 15:
            return Player.WHITE
        elif self.checkers_borne_off(Player.BLACK) == 15:
            return Player.BLACK
        return None
        
    def to_array(self) -> np.ndarray:
        """Convert board to numpy array for neural networks."""
        # Create feature representation for RL agents
        features = np.zeros(198)  # 24*4 + 24*4 + 6 features
        
        # White checkers on each point (4 features per point)
        for i in range(24):
            white_count = max(0, self.points[i])
            if white_count > 0:
                features[i*4] = 1  # Has white checkers
                features[i*4 + 1] = min(white_count, 1)  # 1 checker
                features[i*4 + 2] = max(0, min(white_count - 1, 1))  # 2 checkers
                features[i*4 + 3] = max(0, white_count - 2) / 2  # 3+ checkers (normalized)
                
        # Black checkers on each point (4 features per point)
        for i in range(24):
            black_count = max(0, -self.points[i])
            if black_count > 0:
                features[96 + i*4] = 1  # Has black checkers
                features[96 + i*4 + 1] = min(black_count, 1)  # 1 checker
                features[96 + i*4 + 2] = max(0, min(black_count - 1, 1))  # 2 checkers
                features[96 + i*4 + 3] = max(0, black_count - 2) / 2  # 3+ checkers
                
        # Additional features
        features[192] = self.points[24] / 15  # White on bar (normalized)
        features[193] = -self.points[25] / 15  # Black on bar (normalized)
        features[194] = self.points[26] / 15  # White borne off (normalized)
        features[195] = -self.points[27] / 15  # Black borne off (normalized)
        features[196] = 1 if self.all_checkers_in_home_board(Player.WHITE) else 0
        features[197] = 1 if self.all_checkers_in_home_board(Player.BLACK) else 0
        
        return features
        
    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append("13 14 15 16 17 18   19 20 21 22 23 24")
        
        # Top half
        for row in range(5, -1, -1):
            line = ""
            for point in range(12, 18):
                count = abs(self.points[point])
                if count > row:
                    symbol = "W" if self.points[point] > 0 else "B"
                    line += f" {symbol} "
                else:
                    line += "   "
            line += " | "
            for point in range(18, 24):
                count = abs(self.points[point])
                if count > row:
                    symbol = "W" if self.points[point] > 0 else "B"
                    line += f" {symbol} "
                else:
                    line += "   "
            lines.append(line)
            
        lines.append("=" * 37)
        
        # Bottom half
        for row in range(6):
            line = ""
            for point in range(11, 5, -1):
                count = abs(self.points[point])
                if count > row:
                    symbol = "W" if self.points[point] > 0 else "B"
                    line += f" {symbol} "
                else:
                    line += "   "
            line += " | "
            for point in range(5, -1, -1):
                count = abs(self.points[point])
                if count > row:
                    symbol = "W" if self.points[point] > 0 else "B"
                    line += f" {symbol} "
                else:
                    line += "   "
            lines.append(line)
            
        lines.append("12 11 10  9  8  7    6  5  4  3  2  1")
        
        # Add bar and borne off info
        lines.append(f"White on bar: {self.points[24]}, borne off: {self.points[26]}")
        lines.append(f"Black on bar: {-self.points[25]}, borne off: {-self.points[27]}")
        
        return "\n".join(lines)
