import torch
import numpy as np
from random import choices
from itertools import accumulate
# from chess_primitives import init_board, conjugate_board, candidate_moves, in_check, is_draw, get_played, which_board, evaluate_position
from chess import Board
import chess.svg
import cairosvg
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import copy

def softmax_temp(x, temp=1):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

def entropy(d):
    # Returns the entropy of a discrete distribution
    e = -(d * np.log2(d + 1e-10)).sum() # epsilon value added due to log2(0) == undefined.
    return e

def entropy_temperature(x, target_entropy, T=[1e-3, 1e0, 1e2], tol=1e-3, max_iter=10):
    # returns the temperature parameter required to transform the vector x into a probability distribution with a particular target entropy
    delta = np.inf
    for _ in range(max_iter):
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
            delta = T[2] - T[1]
        else:
            return (T[0]+T[2]) / 2
    return (T[0]+T[2]) / 2

def selector(scores, p=0.3, k=3):
    '''
    This is an elegant idea. Squash the choice distribution to have a target (lower) entropy.
    Selects a token, based on log2(p * len(k)) degrees of freedom.
    '''
    
    if all([score == scores[0] for score in scores]): # If there is no variance in the scores, then just chose randomly.
        return choices(range(len(scores)))[0]
    else:
        # Otherwise target entropy is either proportion p * max_possible_entropy (for small option sets) or as-if k-degree of freedom distribution (for scores >> k)
        target_entropy = min(p * np.log2(len(scores)), np.log2(k))
        # If we abandon the second term above, we allow the model more freedom when there are more options to chose from.
        # Actually we could achieve the same thing by setting k ~ inf. Numpy handles this just fine so np.log2(float('inf')) = inf
        t = entropy_temperature(scores, target_entropy)
        dist = softmax_temp(scores, temp=t)
        return choices(range(len(scores)), cum_weights=list(accumulate(dist)))[0]

class Agent:
    def __init__(self, model=None, p=0.3, k=3):
        self.model, self.p, self.k = model, p, k
        if self.model:
            # assert self.model
            self.model.eval()

    def select_move(self, options):
        # If there is no model passed, then just chose randomly.
        if self.model is None:
            return choices(range(len(options)))[0]
        with torch.no_grad():
            individual_boards = torch.tensor(options).split(1)
            # Score the options with the model
            # scores = self.model(torch.tensor(options))
            scores = [self.model(board) for board in individual_boards]
        # Select end token
        selection = selector(scores, self.p, self.k)
        return selection

def plot_score_panel(white_panel_height):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(1,20))
    fig.set_facecolor('black')
    # Draw black panel
    ax.axvspan(0, 1, color='black', alpha=0.5)
    # Draw white panel over black panel
    ax.axvspan(0, 1, color='white', alpha=1, ymax=white_panel_height)
    ax.axhline(y=0, color='red', linestyle='-', linewidth=10)  # Add a dashed horizontal white line at score 0
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    # Hide axes
    ax.axis('off')
    # Show plot
    plt.tight_layout()
    plt.savefig("evalbar.png")
    plt.close()

def attach_eval_bar(board_path, eval_bar_path, out_path):
    # Load images
    image1 = mpimg.imread(eval_bar_path)
    image2 = mpimg.imread(board_path)
    # Create figure and axes
    # Create figure and gridspec
    fig = plt.figure(figsize=(11, 11), facecolor='black')
    gs = gridspec.GridSpec(1, 2, width_ratios=(1,20))
    # Plot first image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(image1)
    ax0.axis('off')
    # Plot second image
    ax1 = plt.subplot(gs[1])
    ax1.imshow(image2)
    ax1.axis('off')
    # Ensure aspect ratio is equal for both subplots
    ax0.set_aspect('auto')
    ax1.set_aspect('auto')
    # Show the plot
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def play_game(table, agents, max_moves=float('inf'), min_seconds_per_move=2, verbose=False, eval_plot=False):

    # board, color_toplay = starting_state if starting_state is not None else (init_board(play_as='white'), 'white')
    board, color_toplay = init_board(play_as='white'), 'white'
    chs_board = Board()
    chs_boardsvg = chess.svg.board(chs_board, size=600, orientation=chess.WHITE, borders=False, coordinates=False)
    board_png_file_path = f"/mnt/chess/image{table}b.png"
    cairosvg.svg2png(bytestring=chs_boardsvg.encode('utf-8'), write_to="board_temp.png")
    if eval_plot:
        plot_score_panel(0.5)
        attach_eval_bar("board_temp.png", "evalbar.png", board_png_file_path)

    game_result = {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}, 'all_moves': [(board, chs_board, None, chs_boardsvg)]}

    # Play a game until game over.
    while True:

        start = time.perf_counter()

        # Revert any passant pawns to regular pawns if they survived the last turn.
        board[board == 2] = 1
        # Options from each of the starting positions - init as empty dict
        options = candidate_moves(board)
        
        # Check if checkmate or draw.
        player_points, opponent_points, outcome = (None, None, None)
        if len(options) == 0:
            if in_check(board): # Checkmate
                player_points, opponent_points = (-1.0, 1.0)
                outcome = 'Checkmate'

            else: # Stalemate
                player_points, opponent_points = (0.0, 0.0)
                outcome = 'Stalemate'

        if is_draw(board) or len(game_result[color_toplay]['moves']) >= max_moves: # Known draw or max moves reached
            player_points, opponent_points = (0.0, 0.0)
            outcome = 'Draw or timeout'

            if eval_plot:
                if color_toplay == 'white':
                    white_score = evaluate_position(chs_board.fen(), depth_limit=25)
                    if white_score > 0:
                        player_points, opponent_points = (1.0, -1.0)
                    else:
                        player_points, opponent_points = (-1.0, 1.0)
                else:
                    white_score = - evaluate_position(chs_board.fen(), depth_limit=25)
                    if white_score > 0:
                        player_points, opponent_points = (-1.0, 1.0)
                    else:
                        player_points, opponent_points = (1.0, -1.0)
            else:
                # No stockfish evaluation for tiebreak, just a draw.
                player_points, opponent_points = (0.0, 0.0)

        if player_points is not None:
            player, opponent = ('white', 'black') if color_toplay == 'white' else ('black','white')
            game_result[player]['points'] = player_points
            game_result[opponent]['points'] = opponent_points
            if verbose:
                print(f"{outcome} after {len(game_result[color_toplay]['moves'])} moves.")
            return game_result

        move_not_selected = True
        while move_not_selected:

            # Select end_token
            move_selection = agents[color_toplay].select_move(options)
            selected_board = options[move_selection]

            ## GET THE SELECTED MOVE PGN, UPDATE CHS_BOARD FOR RENDERING
            chs_board_pgns = [None] * len(options)
            for cand_move in chs_board.legal_moves:
                # PGN token corresponding to this legal move
                cand_pgn = chs_board.san(cand_move)
                # Pre-process the 
                cand_pgn = cand_pgn.replace('x','').replace('+','').replace('#','')
                target_board = get_played(board, cand_pgn, color_toplay, options)
                cand_ind = which_board(target_board, options)
                # chs_board_fwd = chs_board.copy()
                chs_board_fwd = copy.deepcopy(chs_board)

                move = chs_board_fwd.parse_san(cand_pgn)
                uci_token = move.uci()

                chs_board_fwd.push_san(cand_pgn)
                chs_board_pgns[cand_ind] = (chs_board_fwd, cand_pgn, uci_token)

            chs_board, pgn_token, uci_token = chs_board_pgns[move_selection]
            move = chess.Move.from_uci(uci_token)
            chs_boardsvg = chess.svg.board(chs_board, size=600, orientation=chess.WHITE, lastmove=move, borders=False, coordinates=False)

            if eval_plot:
                # # When evaluating the position at this point, we have just made our move on chs_board
                # # so color_toplay is the opponent.
                copy_for_eval = copy.deepcopy(chs_board)
                if color_toplay == 'white':
                    white_score = - evaluate_position(copy_for_eval.fen(), depth_limit=25)
                else:
                    white_score = evaluate_position(copy_for_eval.fen(), depth_limit=25)
                plot_score_panel((white_score + 10_000) / 20_000)

            move_not_selected = False

        # Move is now selected, chs_board and chs_boardsvg now reflects the updated board state. Send to S3
        cairosvg.svg2png(bytestring=chs_boardsvg.encode('utf-8'), write_to="board_temp.png")
        if eval_plot:
            attach_eval_bar("board_temp.png", "evalbar.png", board_png_file_path)

        # Add this move to the game_record
        game_result[color_toplay]['moves'].append((selected_board, chs_board, pgn_token, chs_boardsvg))
        game_result['all_moves'].append((selected_board, chs_board, pgn_token, chs_boardsvg))

        if verbose:
            print(f"{color_toplay}: {pgn_token}")

        # Swap to opponent's perspective
        color_toplay = 'white' if color_toplay == 'black' else 'black' # Swap to my turn
        board = conjugate_board(selected_board) # Conjugate selected_end_board to opponents perspective

        # Delay next move so that humans can watch!
        move_duration = time.perf_counter() - start
        time_remaining = min_seconds_per_move - move_duration
        if time_remaining > 0:
            time.sleep(time_remaining)

# def set_banner(text="Adam", fontcolor="white", backcolor="black", points=0, winner=False, save_as="black_banner_with_text.png"):
#     # Create a figure and a single subplot
#     fig, ax = plt.subplots(figsize=(12, 2))  # Adjust the size as needed
#     # Set the background color to black
#     fig.set_facecolor(backcolor)
#     # Add the text in white color, centered
#     ax.text(0.35, 0.5, text, color=fontcolor, fontsize=60, ha='right', va='center', fontweight='bold')
#     points_mark = "I" * int(points)
#     if winner:
#         ax.text(0.85, 0.5, f"{points_mark} ♛♚", color=fontcolor, fontsize=100, ha='left', va='center', fontweight='bold')
#     else:
#         ax.text(0.85, 0.5, f"{points_mark}", color=fontcolor, fontsize=100, ha='left', va='center', fontweight='bold')
#     # Remove the axes
#     ax.axis('off')
#     # Save the banner to a file
#     plt.savefig(save_as, bbox_inches='tight', pad_inches=0, dpi=100)
#     # Show the banner
#     plt.close()

def set_banner(text="Adam", points=0, winner=False, save_as="black_banner_with_text.png"):
    fig, ax = plt.subplots(figsize=(12, 2))
    fig.set_facecolor('black')
    ax.axis('off')
    # Define the text position based on the margins
    left_margin, top_margin = 0.05, 0.25
    text_x = left_margin
    text_y = 1 - top_margin
    points_mark = "I" * int(points)
    # Add the text
    ax.text(text_x, text_y, text, color='white', fontsize=80, ha='left', va='top', fontweight='bold')
    if winner:
        ax.text(1 - text_x, text_y, f"{points_mark} ♛♚", color='white', fontsize=80, ha='right', va='top', fontweight='bold')
    else:
        ax.text(1 - text_x, text_y, f"{points_mark}", color='white', fontsize=80, ha='right', va='top', fontweight='bold')
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_as, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

def play_tournament(table, agents, max_games=4, max_moves=float('inf'), min_seconds_per_move=5, verbose=False, eval_plot=False, banner=False):
    # plays a number of paired games, one with agent0 as white, the other with agent0 as black.
    tournament_game_results = []
    is_draw = True

    team_names, agents = zip(*agents.items())
    team0name, team1name = team_names
    agent0, agent1 = agents

    tournament_results = dict()
    tournament_results['agent0'] = 0
    tournament_results['agent1'] = 0

    while is_draw:

        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")
    
        # play game with FIRST model as white
        if banner:
            set_banner(text=team0name, points=tournament_results['agent0'], winner=False, save_as=f"/mnt/chess/image{table}c.png")
            set_banner(text=team1name, points=tournament_results['agent1'], winner=False, save_as=f"/mnt/chess/image{table}a.png")
        kwargs = {'table': table, 'agents': {'white': agent0, 'black': agent1}, 'max_moves': max_moves, 'min_seconds_per_move': min_seconds_per_move, "verbose": verbose, "eval_plot": eval_plot}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)

        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results['agent0'] += game_result['white']['points']
        tournament_results['agent1'] += game_result['black']['points']

        if verbose:
            print(f"\nPlaying Game {len(tournament_game_results) + 1}")

        # play game with SECOND model as white
        if banner:
            set_banner(text=team0name, points=tournament_results['agent0'], winner=False, save_as=f"/mnt/chess/image{table}a.png")
            set_banner(text=team1name, points=tournament_results['agent1'], winner=False, save_as=f"/mnt/chess/image{table}c.png")
        kwargs = {'table': table, 'agents': {'white': agent1, 'black': agent0}, 'max_moves': max_moves, 'min_seconds_per_move': min_seconds_per_move, "verbose": verbose, "eval_plot": eval_plot}
        game_result = play_game(**kwargs)
        tournament_game_results.append(game_result)
        # game_results: {'white': {'moves': [(end_token, end_board), (end_token, end_board), ...], 'points': float}, 'black': {...}}
        tournament_results['agent0'] += game_result['black']['points']
        tournament_results['agent1'] += game_result['white']['points']

        # Check if draw, if so, play again!
        is_draw = tournament_results['agent0'] == tournament_results['agent1']

        if is_draw:
            print("DRAW!")

        # If there's a winner, update their banner with the winning tags
        if not is_draw and banner:
            if tournament_results['agent0'] > tournament_results['agent1']:
                set_banner(text=team0name, points=tournament_results['agent0'], winner=True, save_as=f"/mnt/chess/image{table}a.png")
            else:
                set_banner(text=team1name, points=tournament_results['agent1'], winner=True, save_as=f"/mnt/chess/image{table}c.png")

        # If we've played our max_games, call it a day.
        if len(tournament_game_results) >= max_games:
            # End the loop
            is_draw = False

    return tournament_results, tournament_game_results
