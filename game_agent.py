"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    print('Search timed out')


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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # difference between legal moves for player and opponent
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_diff = float(my_moves - opponent_moves)

    total_cells = game.width * game.height
    centrality = 0
    closeness_score = 0

    # assess how close the player is to the opponent
    a, b = game.get_player_location(player)
    c, d = game.get_player_location(game.get_opponent(player))
    closeness = abs(a-c) + abs(b-d)
    closeness_score = 6*(1/closeness)

    # assess centrality if it is the beginning of the game
    if game.move_count < total_cells/3:
        w, h = game.width / 2., game.height / 2.
        centrality = float((h - a)**2 + (w - b)**2)

    return move_diff + closeness_score + math.sqrt(centrality)


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

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

     # difference between legal moves for player and opponent (the latter given heigher weight)
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_diff = float(my_moves - 2*opponent_moves)

    # assess how close the player is to the opponent
    a, b = game.get_player_location(player)
    c, d = game.get_player_location(game.get_opponent(player))
    closeness = abs(a-c) + abs(b-d)
    closeness_score = 6*(1/closeness)

    return move_diff + closeness_score


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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # difference between legal moves for player and opponent
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    move_diff = float(my_moves - opponent_moves)

    total_cells = game.width * game.height
    centrality = 0

    # assess centrality if it is the beginning of the game
    if game.move_count < total_cells/3:
        w, h = game.width / 2., game.height / 2.
        a, b = game.get_player_location(player)
        centrality = float((h - a)**2 + (w - b)**2)

    return move_diff + math.sqrt(centrality)


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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
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
        best_move = (-1, -1)

        legal_moves = game.get_legal_moves(self)

        if len(legal_moves) == 0:
            return best_move

        if game.move_count == 0 :
            print('1st move')
            return round(game.height/4), round(game.width/4)

        if len(legal_moves) == 1:
            return legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            # catches player 1 timeout and handle player 2 move
           pass

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

        best_move = (-1,-1)
        best_score = float("-inf")
        value = float("-inf")
        #get move list
        legal_moves = game.get_legal_moves()

        if depth == 0 or not legal_moves:
            return best_move

        for move in legal_moves:
            # get min list
            value = self.min_value(game.forecast_move(move), depth - 1)
            #get max value
            if value >= best_score:
                best_move = move
                best_score = value
        #return best move
        return best_move


    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if(depth <= 0 or len(legal_moves) is 0):
            return self.score(game, self)

        best_score = float("-inf")

        for move in legal_moves:
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth-1))
        return best_score

    def min_value(self, game, depth):
         if self.time_left() < self.TIMER_THRESHOLD:
             raise SearchTimeout()

         legal_moves = game.get_legal_moves()
         if(depth <= 0 or len(legal_moves) is 0):
             return self.score(game, self)

         best_score = float("inf")

         for move in legal_moves:
             best_score = min(best_score, self.max_value(game.forecast_move(move), depth-1))

         return best_score


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

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        legal_moves = game.get_legal_moves(self)
        if len(legal_moves) == 0:
            return best_move

        if game.move_count == 0 :
            print('1st move')
            return round(game.height/4), round(game.width/4)

        if len(legal_moves) == 1:
            return legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire
            #Iterative Deepening Searching
            depth = 1
            while True:
                legal_moves = game.get_legal_moves()
                best_move = self.alphabeta(game,depth,7,4)
                depth += 1

        except SearchTimeout:
            # catches player 1 timeout and handle player 2 move
           pass

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

        best_move = (-1,-1)
        best_score = float("-inf")
        value = float("-inf")

        legal_moves = game.get_legal_moves()

        if depth == 0 or not legal_moves:
            return best_move

        for move in legal_moves:
            # get min list
            value = self.min_value(game.forecast_move(move), depth - 1, alpha,beta)
            #set alpha for prunning
            alpha = max(alpha,value)
            #get max value
            if value >= best_score:
                best_move = move
                best_score = value
            #prunning at the top
            if value >= beta:
                return best_move
        #return best move
        return best_move

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if(depth <= 0 or len(legal_moves) is 0):
            return self.score(game,self)

        best_score = float("-inf")

        for move in legal_moves:
            #get max from min
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            if best_score >= beta:
                return best_score
            #set alpha for prunning
            alpha = max(alpha, best_score)

        return best_score

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if(depth <= 0 or len(legal_moves) is 0):
                return self.score(game,self)

        best_score = float("inf")

        for move in legal_moves:
            #get min from max
            best_score = min(best_score,self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if best_score <= alpha:
                return best_score
            #set beta for prunning
            beta = min(beta, best_score)

        return best_score