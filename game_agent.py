"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import time
import random
import math
from sklearn.cluster import KMeans, DBSCAN
from isolation import Board

INVALID_MOVE = (-1, -1)
    
class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    score = 0
    offset = [-0.01, -0.02, -0.01, 0.04, -0.05, -0.02, -0.07, -0.05, 0.03, 0.09, 0.07, -0.03, 0.00, -0.02, 0.08, 0.04, 0.03, 0.02, 0.05, 0.08, 0.01, 0.07, 0.06, 0.12, 0.19, 0.03, 0.05, -0.04, 0.03, 0.07, 0.04, 0.08, 0.09, -0.02, -0.06, -0.10, -0.02, 0.11, -0.03, -0.02, -0.03, -0.07, 0.01, -0.07, -0.04, 0.04, -0.04, -0.07, -0.12]

    current_move = game.get_player_location(player)
    index = current_move[0] * 7 + current_move[1]
    scr = custom_score_2(game, player)
    return scr + 16 * offset[index]

def custom_score(game, player):
    offset = [
        -0.01, -0.02, -0.01, 0.04, -0.05, -0.02, -0.07, 
        -0.05, 0.03, 0.09, 0.07, -0.03, 0.00, -0.02, 
        0.08, 0.04, 0.03, 0.02, 0.05, 0.08, 0.01, 
        0.07, 0.06, 0.12, 0.19, 0.03, 0.05, -0.04, 
        0.03, 0.07, 0.04, 0.08, 0.09, -0.02, -0.06, 
        -0.10, -0.02, 0.11, -0.03, -0.02, -0.03, -0.07, 
        0.01, -0.07, -0.04, 0.04, -0.04, -0.07, -0.12]

    current_move = game.get_player_location(player)
    index = current_move[0] * 7 + current_move[1]
    scr = custom_score_2(game, player)
    return scr + 16 * offset[index]

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    def get_legal_moves(game, index):
        c = index // game.height
        r = index % game.height
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        valid_moves = [(r + dr, c + dc) for dr, dc in directions
                       if game.move_is_legal((r + dr, c + dc))]
        return valid_moves

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_index = 0
    opponent_index = 0
    if game._player_1 == player:
        player_index = game._board_state[-1]
        opponent_index = game._board_state[-2]
    else:
        opponent_index = game._board_state[-1]
        player_index = game._board_state[-2]

    own_moves = len(get_legal_moves(game, player_index))
    opp_moves = len(get_legal_moves(game, opponent_index))

    return float(own_moves - opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    score = custom_score_2(game, player)
    occupied_spaces = get_occupied_spaces(game)
    d = DBSCAN(eps=2).fit(occupied_spaces)
    occupied_spaces_without_outliers = [occupied_spaces[index] for index, el in enumerate(d.labels_) if el != -1]
    if len(occupied_spaces_without_outliers) < 8:
        return score

    kmeans_score = custom_score_kmeans(game, player, occupied_spaces_without_outliers)

    return score + kmeans_score

def custom_score_kmeans(game, player, occupied_spaces):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w,h = KMeans(n_clusters=1, random_state=0).fit(occupied_spaces).cluster_centers_[0]
    corner_w = 0
    corner_h = 0
    if w > game.width / 2:
        corner_w = game.width
    else:
        corner_w = 0
    
    if h > game.height / 2:
        corner_h = game.height
    else:
        corner_h = 0

    row, column = game.get_player_location(game.get_opponent(player))
    return math.sqrt(float((corner_h - row)**2 + (corner_w - column)**2)) * -1

def get_occupied_spaces(game):
        """Return a list of the locations that are still available on the board.
        """
        return [[i, j] for j in range(game.width) for i in range(game.height)
                if game._board_state[i + j * game.height] != Board.BLANK]


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=1, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = INVALID_MOVE

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.max_value(game, depth)[1]

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self), INVALID_MOVE

        best_move = INVALID_MOVE
        max_val = float("-inf")
        for move in game.get_legal_moves(self):
            value, _ = self.min_value(game.forecast_move(move), depth-1)
            if value > max_val:
                max_val = value
                best_move = move
        
        return max_val, best_move

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self), INVALID_MOVE

        best_move = INVALID_MOVE
        min_val = float("inf")
        for move in game.get_legal_moves(game.get_opponent(self)):
            value, _ = self.max_value(game.forecast_move(move), depth-1)
            if value < min_val:
                min_val = value
                best_move = move

        return min_val, best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        legal_moves = game.get_legal_moves(self)
        if not len(legal_moves):
            return INVALID_MOVE

        # first move
        if not game.get_player_location(self):
            center = math.floor(game.width / 2.), math.floor(game.height / 2.)
            if game.move_is_legal(center):
                return center
            else:
                if game.width > game.height:
                    return center[0] + 1, center[1]
                else:
                    return center[0], center[1] + 1

        best_move = legal_moves[0]
        depth = 1

        while True:
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1    
            except SearchTimeout:
                return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves(self)) == 0:
            return INVALID_MOVE

        legal_moves = game.get_legal_moves(self)
        best_move = legal_moves[0]
        for move in legal_moves:
            value, _ = self.min_value(game.forecast_move(move), 1, depth-1, alpha, beta)
            if value > alpha:
                alpha = value
                best_move = move
            
        return best_move

    def max_value(self, game, current_depth, depth_left, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth_left == 0 or not len(game.get_legal_moves()):
            return self.score(game, self), INVALID_MOVE
                
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0]
        for move in legal_moves:
            value, _ = self.min_value(game.forecast_move(move), current_depth+1, depth_left-1, alpha, beta)
            if value >= beta:
                return value, move
            if value > alpha:
                alpha = value
                best_move = move
        
        return alpha, best_move

    def min_value(self, game, current_depth, depth_left, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth_left == 0 or not len(game.get_legal_moves()):
            return self.score(game, self), INVALID_MOVE

        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0]
        for move in legal_moves:
            value, _ = self.max_value(game.forecast_move(move), current_depth+1, depth_left-1, alpha, beta)
            if value <= alpha:
                return value, move
            if value < beta:
                beta = value
                best_move = move

        return beta, best_move
