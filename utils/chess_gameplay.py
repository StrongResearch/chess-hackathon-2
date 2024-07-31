import torch
import torch.nn as nn
import numpy as np
from random import choice, choices
from itertools import accumulate
from utils.chess_primitives import init_board, conjugate_board, candidate_moves, in_check, is_draw, get_played, which_board, evaluate_position
from chess import Board
import chess.svg
import cairosvg
import time
import copy
import os

def softmax_temp(x, temp=1):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

def entropy(d):
    # Returns the entropy of a discrete distribution
    e = -(d * np.log2(d + 1e-10)).sum() # epsilon value added due to log2(0) == undefined.
    return e

def entropy_temperature(x, target_entropy, T=[1e-3, 1e0, 1e2], tol=1e-3, max_iter=10_000):
    # returns the temperature parameter (to within tol) required to transform the vector x into a 
    # probability distribution with a particular target entropy
    delta = np.inf
    for i in range(max_iter):
        if delta > tol:
            E = [entropy(softmax_temp(x, temp=t)) for t in T]
            if E[0] > target_entropy:
                T = [T[0]/2, T[1], T[2]]
            elif E[2] < target_entropy:
                T = [T[0], T[1], T[2]*2]
            elif E[0] < target_entropy < E[1]:
                T = [T[0], (T[0]+T[1])/2, T[1]]
            elif E[1] < target_entropy < E[2]:
                T = [T[1], (T[1]+T[2])/2, T[2]]
            delta = (E[2] - E[0]) / target_entropy
        else:
            return (T[0]+T[2]) / 2
    print("WARNING: Entropy search depth exceeded.")
    return (T[0]+T[2]) / 2

def sans_to_pgn(move_sans):
    move = 1
    pgn = ["1."]
    for i,san in enumerate(move_sans, start=1):
        pgn += [san, " "]
        if i % 2 == 0:
            pgn.append(f"{int((i+2)/2)}.")
    return ''.join(pgn)

def selector(scores, p=0.3, k=3):
    '''
    Squashes the options distribution to have a target (lower) entropy.
    Selects a token, based on log2(p * len(k)) degrees of freedom.
    '''
    # If there is no variance in the scores, then just chose randomly.
    if all([score == scores[0] for score in scores]): 
        return choice(range(len(scores)))
    else:
        # Otherwise target entropy is either proportion p * max_possible_entropy (for small option sets) or 
        # as-if k-degree of freedom distribution (for num_scores >> k)
        target_entropy = min(p * np.log2(len(scores)), np.log2(k))
        # If we abandon the second term above, we allow the model more freedom when there are more options to 
        # chose from. Actually we could achieve the same thing by setting k ~ inf. Numpy handles this just fine 
        # so np.log2(float('inf')) = inf
        t = entropy_temperature(scores, target_entropy)
        dist = softmax_temp(scores, temp=t)
        return choices(range(len(scores)), cum_weights=list(accumulate(dist)))[0]

class Agent:
    def __init__(self, model=None, p=0.3, k=3):
        self.model, self.p, self.k = model, p, k

        if self.model:
            assert isinstance(model, nn.Module), "ERROR: model must be a torch nn.Module"
            self.model.eval()

    def select_move(self, pgn, legal_moves):
        # If there is no model passed, then just chose randomly.
        if self.model is None:
            return choice(legal_moves)

        scores = []
        with torch.no_grad():
            for move in legal_moves:
                score = self.model.score(pgn, move)
                scores.append(score)

        # Index of selected move
        selection = selector(np.array(scores), self.p, self.k)
        return legal_moves[selection]

def play_game(table, agents, max_moves=float('inf'), min_seconds_per_move=2, verbose=False, poseval=False, image_path="/mnt/chess/"):

    board = Board()
    color_toplay = 'white'
    move_sans = [] # for constructing the pgn
    board_svg = chess.svg.board(board, size=600, orientation=chess.WHITE, borders=False, coordinates=False)
    board_png_file_path = os.path.join(image_path, f"image{table}b.png")
    cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), write_to=board_png_file_path)
    game_result = {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}, 'all_moves': [(board, None, board_svg)]}

    # Play a game until game over.
    while True:

        start = time.perf_counter()
        whites_turn = board.turn
        
        # Check if checkmate or draw.
        player_points, opponent_points = (None, None)

        checkmate = board.is_checkmate()
        draw = board.is_variant_draw()
        stalemate = board.is_stalemate()

        if checkmate:
            player_points, opponent_points = (-1.0, 1.0)

        elif draw or stalemate:
            player_points, opponent_points = (0.0, 0.0)

        elif len(game_result[color_toplay]['moves']) >= max_moves:
            if poseval:
                score = evaluate_position(board.fen(), depth_limit=25)
                player_points, opponent_points = (score, -score)
            else:
                player_points, opponent_points = (0.0, 0.0)

        if player_points is not None:
            player, opponent = ('white', 'black') if whites_turn else ('black','white')
            game_result[player]['points'] = player_points
            game_result[opponent]['points'] = opponent_points
            return game_result

        # generate legal move sans
        legal_moves = list(board.legal_moves)
        legal_move_sans = [board.san(move) for move in legal_moves]

        # selected move
        pgn = sans_to_pgn(move_sans)
        selected_move_san = agents[color_toplay].select_move(pgn, legal_move_sans)
        selected_move = legal_moves[legal_move_sans.index(selected_move_san)]
        move_sans.append(selected_move_san)

        # push move to the board
        board.push_san(selected_move_san)

        # print the new board to SVG
        board_svg = chess.svg.board(board, size=600, orientation=chess.WHITE, lastmove=selected_move, borders=False, coordinates=False)
        cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), write_to=board_png_file_path)

        # Add this move to the game_record
        game_result[color_toplay]['moves'].append((board, selected_move_san, board_svg))
        game_result['all_moves'].append((board, selected_move_san, board_svg))

        if verbose:
            print(f"{color_toplay}: {selected_move_san}")

        # Swap to opponent's perspective
        color_toplay = 'white' if color_toplay == 'black' else 'black' # Swap to my turn

        # Delay next move so that humans can watch!
        move_duration = time.perf_counter() - start
        time_remaining = min_seconds_per_move - move_duration
        if time_remaining > 0:
            time.sleep(time_remaining)

def play_tournament(table, agents, max_games=4, max_moves=float('inf'), min_seconds_per_move=5, verbose=False, poseval=False, image_path="/mnt/chess/"):
    # plays a number of paired games, one with agent0 as white, the other with agent0 as black.
    tournament_game_results = []
    is_draw = True

    tournament_results = dict()
    tournament_results['agent0'] = 0
    tournament_results['agent1'] = 0

    kwargs = {
        "tabl": table, "max_moves": max_moves, "min_seconds_per_move": min_seconds_per_move, 
        "verbose": verbose, "poseval": poseval, "image_path": image_path
    }

    while is_draw:
        
        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")
    
        # play game with FIRST model as white
        kwargs["agents"] = {'white': agents[0], 'black': agents[1]}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)
        tournament_results['agent0'] += game_result['white']['points']
        tournament_results['agent1'] += game_result['black']['points']

        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")

        # play game with SECOND model as white
        kwargs["agents"] = {'white': agents[1], 'black': agents[0]}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)
        tournament_results['agent0'] += game_result['black']['points']
        tournament_results['agent1'] += game_result['white']['points']

        # Check if draw, if so, play again!
        is_draw = tournament_results['agent0'] == tournament_results['agent1']

        if is_draw:
            print("DRAW!")

        # If we've played our max_games, call it a day.
        if len(tournament_game_results) >= max_games:
            # End the loop
            is_draw = False

    return tournament_results, tournament_game_results
