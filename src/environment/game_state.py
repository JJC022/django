"""Game state management for backgammon."""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .board import Board, Player


@dataclass
class Move:
    """Represents a single checker move."""
    from_point: int
    to_point: int
    die_value: int


@dataclass
class Turn:
    """Represents a complete turn with all moves."""
    moves: List[Move]
    dice: Tuple[int, int]
    player: Player


class GameState:
    """Manages the complete game state including board, dice, and turn logic."""
    
    def __init__(self):
        """Initialize a new game state."""
        self.board = Board()
        self.current_player = Player.WHITE
        self.dice = (0, 0)
        self.available_moves = []
        self.turn_history = []
        self.game_over = False
        self.winner = None
        
    def roll_dice(self) -> Tuple[int, int]:
        """Roll two dice and return the values."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        self.dice = (die1, die2)
        return self.dice
        
    def get_available_dice_values(self) -> List[int]:
        """Get available dice values for moves."""
        if self.dice[0] == self.dice[1]:
            # Doubles: use each die value 4 times
            return [self.dice[0]] * 4
        else:
            return list(self.dice)
            
    def generate_legal_moves(self, player: Player, dice_values: List[int]) -> List[List[Move]]:
        """Generate all legal move sequences for the current position."""
        legal_moves = []
        
        # If player has checkers on bar, must enter them first
        if self.board.checkers_on_bar(player) > 0:
            legal_moves = self._generate_moves_from_bar(player, dice_values)
        else:
            legal_moves = self._generate_regular_moves(player, dice_values)
            
        return legal_moves
        
    def _generate_moves_from_bar(self, player: Player, dice_values: List[int]) -> List[List[Move]]:
        """Generate moves when player has checkers on the bar."""
        legal_moves = []
        
        # Try to enter checkers from bar
        bar_point = 24 if player == Player.WHITE else 25
        home_start = 18 if player == Player.WHITE else 0
        direction = 1 if player == Player.WHITE else -1
        
        for die_value in dice_values:
            entry_point = home_start + (die_value - 1) * direction
            if player == Player.BLACK:
                entry_point = 24 - die_value
                
            if self.board.can_move_to(entry_point, player):
                # Create a move from bar
                move = Move(bar_point, entry_point, die_value)
                
                # Try this move and see what other moves are possible
                temp_board = self.board.copy()
                temp_board.move_checker(bar_point, entry_point, player)
                
                remaining_dice = dice_values.copy()
                remaining_dice.remove(die_value)
                
                if remaining_dice and temp_board.checkers_on_bar(player) == 0:
                    # Can make additional moves
                    sub_moves = self._generate_regular_moves_recursive(
                        temp_board, player, remaining_dice, [move]
                    )
                    legal_moves.extend(sub_moves)
                else:
                    legal_moves.append([move])
                    
        return legal_moves
        
    def _generate_regular_moves(self, player: Player, dice_values: List[int]) -> List[List[Move]]:
        """Generate regular moves (not from bar)."""
        return self._generate_regular_moves_recursive(self.board, player, dice_values, [])
        
    def _generate_regular_moves_recursive(self, board: Board, player: Player, 
                                        dice_values: List[int], current_moves: List[Move]) -> List[List[Move]]:
        """Recursively generate all possible move combinations."""
        if not dice_values:
            return [current_moves] if current_moves else [[]]
            
        all_moves = []
        direction = 1 if player == Player.WHITE else -1
        
        # Try each available die value
        for i, die_value in enumerate(dice_values):
            remaining_dice = dice_values.copy()
            remaining_dice.pop(i)
            
            # Try moving from each point
            for from_point in range(24):
                if not board.can_move_from(from_point, player):
                    continue
                    
                to_point = from_point + die_value * direction
                
                # Check for bearing off
                if board.all_checkers_in_home_board(player):
                    home_end = 26 if player == Player.WHITE else 27
                    if (player == Player.WHITE and from_point >= 18 and to_point >= 24) or \
                       (player == Player.BLACK and from_point <= 5 and to_point < 0):
                        # Bearing off
                        if to_point == 24 and player == Player.WHITE:
                            to_point = 26
                        elif to_point < 0 and player == Player.BLACK:
                            to_point = 27
                        else:
                            # Check if this is the furthest checker for exact bear off
                            if not self._is_exact_bear_off_legal(board, player, from_point, die_value):
                                continue
                            to_point = home_end
                            
                # Regular move
                if 0 <= to_point <= 23 and not board.can_move_to(to_point, player):
                    continue
                    
                # Make the move on a temporary board
                temp_board = board.copy()
                if temp_board.move_checker(from_point, to_point, player):
                    move = Move(from_point, to_point, die_value)
                    new_moves = current_moves + [move]
                    
                    # Recursively try remaining dice
                    sub_moves = self._generate_regular_moves_recursive(
                        temp_board, player, remaining_dice, new_moves
                    )
                    all_moves.extend(sub_moves)
                    
        # If no moves were possible, return current moves
        if not all_moves and current_moves:
            all_moves.append(current_moves)
        elif not all_moves and not current_moves:
            all_moves.append([])  # No moves possible
            
        return all_moves
        
    def _is_exact_bear_off_legal(self, board: Board, player: Player, from_point: int, die_value: int) -> bool:
        """Check if bearing off with higher die than needed is legal."""
        if player == Player.WHITE:
            # Check if there are checkers on higher points
            for point in range(from_point + 1, 24):
                if board.points[point] > 0:
                    return False
        else:
            # Check if there are checkers on higher points (lower numbers for black)
            for point in range(0, from_point):
                if board.points[point] < 0:
                    return False
        return True
        
    def make_move_sequence(self, moves: List[Move]) -> bool:
        """Execute a sequence of moves."""
        if not moves:
            return True
            
        # Validate that all moves use available dice
        dice_values = self.get_available_dice_values()
        used_dice = [move.die_value for move in moves]
        
        for die_value in used_dice:
            if die_value not in dice_values:
                return False
            dice_values.remove(die_value)
            
        # Execute moves
        for move in moves:
            if not self.board.move_checker(move.from_point, move.to_point, self.current_player):
                return False
                
        # Record the turn
        turn = Turn(moves, self.dice, self.current_player)
        self.turn_history.append(turn)
        
        # Check for game over
        if self.board.is_game_over():
            self.game_over = True
            self.winner = self.board.get_winner()
        else:
            # Switch players
            self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
            
        return True
        
    def get_state_vector(self) -> List[float]:
        """Get state representation for RL agents."""
        features = self.board.to_array().tolist()
        
        # Add current player
        features.append(1.0 if self.current_player == Player.WHITE else -1.0)
        
        # Add dice information (if rolled)
        features.extend([self.dice[0] / 6.0, self.dice[1] / 6.0])
        
        return features
        
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        new_state = GameState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.dice = self.dice
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.turn_history = self.turn_history.copy()
        return new_state
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = Board()
        self.current_player = Player.WHITE
        self.dice = (0, 0)
        self.available_moves = []
        self.turn_history = []
        self.game_over = False
        self.winner = None
