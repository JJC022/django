"""Game utility functions for backgammon."""

import numpy as np
from typing import List, Tuple, Optional
from ..environment.board import Board, Player
from ..environment.game_state import GameState, Move


class GameUtils:
    """Utility functions for game analysis and manipulation."""
    
    @staticmethod
    def pip_count(board: Board, player: Player) -> int:
        """Calculate pip count for a player.
        
        The pip count is the total number of points a player needs to move
        all checkers to bear them off.
        
        Args:
            board: Game board
            player: Player to calculate for
            
        Returns:
            Total pip count
        """
        pip_count = 0
        
        for point in range(24):
            checker_count = board.get_checker_count(point, player)
            if checker_count > 0:
                if player == Player.WHITE:
                    # White moves from low to high points
                    distance_to_home = 24 - point
                else:
                    # Black moves from high to low points
                    distance_to_home = point + 1
                    
                pip_count += checker_count * distance_to_home
                
        # Add checkers on bar (they need to travel the full board)
        bar_checkers = board.checkers_on_bar(player)
        pip_count += bar_checkers * 25
        
        return pip_count
    
    @staticmethod
    def race_position(board: Board) -> bool:
        """Check if the position is a race (no contact between players).
        
        Args:
            board: Game board
            
        Returns:
            True if it's a race position
        """
        # Find the most advanced checker for each player
        white_most_advanced = -1
        black_most_advanced = 24
        
        for point in range(24):
            if board.points[point] > 0:  # White checker
                white_most_advanced = max(white_most_advanced, point)
            if board.points[point] < 0:  # Black checker
                black_most_advanced = min(black_most_advanced, point)
                
        # If white's most advanced checker is ahead of black's most advanced,
        # it's a race
        return white_most_advanced > black_most_advanced
    
    @staticmethod
    def blots_count(board: Board, player: Player) -> int:
        """Count the number of blots (single checkers) for a player.
        
        Args:
            board: Game board
            player: Player to count for
            
        Returns:
            Number of blots
        """
        blots = 0
        for point in range(24):
            if board.get_checker_count(point, player) == 1:
                blots += 1
        return blots
    
    @staticmethod
    def blocks_count(board: Board, player: Player) -> int:
        """Count consecutive blocked points (prime building).
        
        Args:
            board: Game board
            player: Player to count for
            
        Returns:
            Length of longest consecutive block
        """
        max_block = 0
        current_block = 0
        
        # Check points in the direction of movement
        if player == Player.WHITE:
            points_to_check = range(24)
        else:
            points_to_check = range(23, -1, -1)
            
        for point in points_to_check:
            if board.get_checker_count(point, player) >= 2:
                current_block += 1
                max_block = max(max_block, current_block)
            else:
                current_block = 0
                
        return max_block
    
    @staticmethod
    def bearing_off_efficiency(board: Board, player: Player) -> float:
        """Calculate bearing off efficiency (0-1).
        
        Args:
            board: Game board
            player: Player to calculate for
            
        Returns:
            Efficiency score (higher is better)
        """
        if not board.all_checkers_in_home_board(player):
            return 0.0
            
        home_points = range(18, 24) if player == Player.WHITE else range(6)
        total_checkers = 0
        weighted_position = 0
        
        for i, point in enumerate(home_points):
            checkers = board.get_checker_count(point, player)
            total_checkers += checkers
            # Weight by distance from bearing off (closer is better)
            weight = 6 - i if player == Player.WHITE else i + 1
            weighted_position += checkers * weight
            
        if total_checkers == 0:
            return 1.0  # All borne off
            
        # Normalize by ideal position (all checkers on the 6-point)
        ideal_position = total_checkers * 6
        return weighted_position / ideal_position if ideal_position > 0 else 0.0
    
    @staticmethod
    def position_class(board: Board) -> str:
        """Classify the type of position.
        
        Args:
            board: Game board
            
        Returns:
            Position classification string
        """
        if GameUtils.race_position(board):
            return "race"
        elif board.checkers_on_bar(Player.WHITE) > 0 or board.checkers_on_bar(Player.BLACK) > 0:
            return "contact_with_bar"
        elif board.all_checkers_in_home_board(Player.WHITE) or board.all_checkers_in_home_board(Player.BLACK):
            return "bearing_off"
        else:
            return "contact"
    
    @staticmethod
    def evaluate_position_features(game_state: GameState, player: Player) -> np.ndarray:
        """Extract high-level position features for evaluation.
        
        Args:
            game_state: Current game state
            player: Player perspective
            
        Returns:
            Feature vector
        """
        board = game_state.board
        opponent = Player.BLACK if player == Player.WHITE else Player.WHITE
        
        features = []
        
        # Basic counts
        features.append(GameUtils.pip_count(board, player))
        features.append(GameUtils.pip_count(board, opponent))
        features.append(GameUtils.blots_count(board, player))
        features.append(GameUtils.blots_count(board, opponent))
        features.append(GameUtils.blocks_count(board, player))
        features.append(GameUtils.blocks_count(board, opponent))
        
        # Position type indicators
        features.append(1.0 if GameUtils.race_position(board) else 0.0)
        features.append(1.0 if board.all_checkers_in_home_board(player) else 0.0)
        features.append(1.0 if board.all_checkers_in_home_board(opponent) else 0.0)
        
        # Bar and bearing off
        features.append(board.checkers_on_bar(player))
        features.append(board.checkers_on_bar(opponent))
        features.append(board.checkers_borne_off(player))
        features.append(board.checkers_borne_off(opponent))
        
        # Efficiency measures
        features.append(GameUtils.bearing_off_efficiency(board, player))
        features.append(GameUtils.bearing_off_efficiency(board, opponent))
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def move_to_string(move: Move) -> str:
        """Convert a move to readable string format.
        
        Args:
            move: Move object
            
        Returns:
            String representation
        """
        if move.to_point == 26:
            return f"{move.from_point + 1}/off"
        elif move.to_point == 27:
            return f"{move.from_point + 1}/off"
        elif move.from_point == 24:
            return f"bar/{move.to_point + 1}"
        elif move.from_point == 25:
            return f"bar/{move.to_point + 1}"
        else:
            return f"{move.from_point + 1}/{move.to_point + 1}"
    
    @staticmethod
    def moves_to_string(moves: List[Move]) -> str:
        """Convert a list of moves to readable string format.
        
        Args:
            moves: List of moves
            
        Returns:
            String representation
        """
        if not moves:
            return "No moves"
        return " ".join(GameUtils.move_to_string(move) for move in moves)
