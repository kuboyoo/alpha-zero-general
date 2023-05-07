import numpy as np
import os

from MCTS import MCTS
from splendor.SplendorGame import SplendorGame as Game
from splendor.NNet import NNetWrapper as NNet
import Arena
from utils import *
from pit import HumanPlayer
from pathlib import Path
import argparse

def create_player(name, args, game):
	num_players = len(args.players)
	print("Num of Players: ", num_players)

	if name == 'human':
		return HumanPlayer(game).play
	
	elif name == "ai_1": #読み手数ごとの強さ比較用
		num_mcts = args.numMCTSSims1
		name = "./splendor/pretrained_2players.pt"
		bias = None

	elif name == "ai_2":
		num_mcts = args.numMCTSSims2
		name = "./splendor/pretrained_2players.pt"
		bias = None
	
	else:
		bias = None
		num_mcts = args.numMCTSSims

	# set default values but will be overloaded when loading checkpoint
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
	mcts_args = dotdict({
		'numMCTSSims'     : num_mcts, # if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
		'cpuct'           : args.cpuct       if args.cpuct       else additional_keys.get('cpuct'      , 1.0),
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
		'no_mem_optim'    : False,
	})
	mcts = MCTS(game, net, mcts_args)
	player = lambda x: np.argmax(mcts.getActionProb(x, temp=0, force_full_search=True, bias=bias)[0])
	return player

def play(args, game, curPlayer, board):
	players = [p+'/best.pt' if os.path.isdir(p) else p for p in args.players]
	num_players = len(players)

	if num_players == 2:
		print(players[0], 'vs', players[1])
		player1, player2 = create_player(players[0], args, game), create_player(players[1], args, game)
		human = 'human' in players
		arena = Arena.Arena(player1, player2, None, game, args, display=game.printBoard, no_record=True)
		result = arena.playGames(args.num_games, verbose=args.display or human, cur_player=curPlayer, board=board)
	elif num_players == 3:
		print(players[0], 'vs', players[1], 'vs', players[2])
		player1 = create_player(players[0], args, game)
		player2 = create_player(players[1], args, game)
		player3 = create_player(players[2], args, game)
		human = 'human' in players
		arena = Arena.Arena(player1, player2, player3, game, args, display=game.printBoard, no_record=True)
		result = arena.playGames(1, verbose=True or human, cur_player=curPlayer, board=board)
	return result

#任意の手番から再開
def restart(game, args, curPlayer, board):
	#elif args.reference or len(args.players) > 2: #editted
  if len(args.players) >= 2:
    play(args, game, curPlayer, board)
  else:
    raise Exception('Please specify a player (ai folder, random, greedy or human)')


def main(args):
  turn_num = args.turn_num
  num_mcts = args.numMCTSSims
  record_dir = Path("./record/220217_01/")

  board_path = record_dir.joinpath("board_turn_%d.pkl" % (turn_num+1))
  board = loadPkl(board_path)
  num_players = 3
  curPlayer = turn_num % num_players

  model_name = "./splendor/pretrained_%dplayers.pt" % num_players
  game = Game(num_players)
  
  game.printBoard(board)
  canonical_board = game.getCanonicalForm(board, curPlayer)
  print(f"Player {curPlayer} 's. turn ...")
  print("Num of MCTS: ", )
  restart(game, args, curPlayer, board)
  
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='tester')
  parser.add_argument('--turn-num', '-n' , action='store', default=0, type=int, help='review target number(1-)')
  parser.add_argument('--numMCTSSims','-m' , action='store', default=None  , type=int  , help='Number of games moves for MCTS to simulate.')
  parser.add_argument('--players', '-p' , metavar='player', nargs='*', help='list of players to test (either file, or "human" or "random")')
  parser.add_argument('--cpuct'              , '-c' , action='store', default=None  , type=float, help='')
  args = parser.parse_args()
  main(args)