import numpy as np
import random
import time
from copy import deepcopy
from typing import Any
from operator import attrgetter
#from numba import njit
#import numba

from .SplendorLogic import print_board, move_to_str
from .SplendorGame import Game
from .SplendorGame import Board

#for AlphaBetaSearch
MAX_SEARCH_TIME = 10 #sec
DEFAULT_DEPTH = 6 #2player => 9 / 3player => 13 best?

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board, player=0):
		valids = self.game.getValidMoves(board, player)
		action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int), k=1)[0]
		return action



class HumanPlayer():
	def __init__(self, game):
		self.game = game

	def show_all_moves(self, valid):
		n = 0
		for i, v in enumerate(valid):
			if i in [12,12+15,12+15+3+30]:
				print()
			if v:
				n += 1
				print(f'{i} = {move_to_str(i, short=True)}', end='   ')
				if n % 10 == 0:
					print('\n')
		print()

	def show_main_moves(self, valid):
		# Number of max gems that player can take
		if any(valid[45:55]):
			can_take = 3
		elif any(valid[35:45]):
			can_take = 2
		elif any(valid[30:35]):
			can_take = 1
		else:
			can_take = 0
		need_to_give_gems = (self.game.board.players_gems[0].sum() >= 9)

		print()
		if any(valid[12:27]):
			print(f'12-26 = rsv', end='   ')
		for i, v in enumerate(valid):
			if v:
				if 0<=i<12 or 27<=i<30 or (30<=i<35 and can_take<=1) or (35<=i<45 and can_take<=2) or 45<=i<60 or (60<=i<80 and need_to_give_gems):
					print(f'{i} = {move_to_str(i, short=True)}', end='   ')		
		print('(+ to show all moves)')

	def play(self, board):
		# print_board(self.game.board)
		valid = self.game.getValidMoves(board, 0)
		self.show_main_moves(valid)
		while True:
			input_move = input()
			if input_move == '+':
				self.show_all_moves(valid)
			else:
				try:
					a = int(input_move)
					if not valid[a]:
						raise Exception('')
					break
				except:
					print('Invalid move:', input_move)
		return a


class GreedyPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		valids = self.game.getValidMoves(board, 0)
		candidates = []
		initial_score = self.game.getScore(board, 0)
		for m in [m_ for m_, v in enumerate(valids) if v>0]:
			nextBoard, _ = self.game.getNextState(board, 0, m)
			score = self.game.getScore(nextBoard, 1)
			candidates += [(score, m)]
		max_score = max(candidates, key=lambda x: x[0])[0]
		if max_score == initial_score:
			actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0 and 0      <=m<12        ]
			if len(actions_leading_to_max) == 0:
				actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0 and 12+15+3<=m<12+15+3+30]
				if len(actions_leading_to_max) == 0:
					actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0]
		else:
			actions_leading_to_max = [m for (s,m) in candidates if s==max_score]
		move = random.choice(actions_leading_to_max)
		return move



class Node:
	def __init__(self, label, player, board, action, expect):
		self.label = label #盤面ID
		self.player = player
		self.board = board
		self.action = action
		self.expect = expect #その行動によって加算されるポイント
	
	def get_children(self, game, game_board, board, player):
		#canonical_board = game.getCanonicalForm(game_board, player)
		children = []
		valids = game_board.valid_moves(player)
		valids = np.where(valids == True)[0][:-1] #do nothingは回避

		board_ = deepcopy(board)
		prev_point = game_board.get_score(player)

		is_empty_gold = (game_board.bank[0][5] == 0)
		is_full = (np.sum(game_board.players_gems[player][:6]) == 10)

		for action in valids:
			#金が枯れている場合 or 手持ちチップ10枚 => 予約は候補手から除外
			if (is_empty_gold or is_full) and (action < 12+15 and action >= 12):
				continue

			#game_board_ = game_board_prev
			game_board_ = Board(game.num_players)
			game_board_.copy_state(board_, True)
			next_player = game_board_.make_move(action, player, True)

			next_point = game_board_.get_score(player)
			expect = next_point - prev_point #その行動によって加算されるポイント

			#if next_player != 0:
			#	game_board_.swap_players(next_player)
			state_id = game.stringRepresentation(board_) #ハッシュ for 同一盤面除去見据えて
			#state_id = 0
			children.append(Node(state_id, next_player, game_board_, action, expect))
		children.sort(key=attrgetter('expect'), reverse=True)
		return children
		

class AlphaBetaPlayer():
	def __init__(
		self,
		game: Game,
		player: int,
		mcts,
		depth: int = DEFAULT_DEPTH
	):
		self.game = game
		self.player = player
		self.depth = depth
		self.num_player = self.game.getNumberOfPlayers()
		self.mcts = mcts


	#盤面評価値
	def valueFuncNN(self, canonical_board, cur_player):
		v = self.mcts.search(canonical_board, False, False)
		#print("value: ", v)
		sign = -1 if cur_player != self.player else 1
		return v[cur_player] * sign

	
	def valueFunc(self, canonical_board, cur_player, depth):
		val_total = 0.
		
		for player in range(self.num_player):
			sum_paid = 0.

			player_gems = canonical_board.players_gems[player][:5]
			player_cards = canonical_board.players_cards[player][:5]
			
			#print("valid_buy: ", np.count_nonzero(canonical_board._valid_buy(player)))
			buyable_bool_board = canonical_board._valid_buy(player)
			buyable_bool_reserve = canonical_board._valid_buy_reserve(player)

			#print("buyable_bool_board: ", buyable_bool_board)
			#print("buyable_bool_reserve: ", buyable_bool_reserve)

			buyable_id = np.where(buyable_bool_board > 0)[0].tolist() + (np.where(buyable_bool_reserve)[0] + 12).tolist()
			#print("buyable_id: ",buyable_id)
			buyable_gain = 0.
			buyable_gains = []

			cards = canonical_board.cards_tiers.tolist() + canonical_board.players_reserved.tolist()
			
			for i, card in enumerate(cards):
				card_cost = card[:5]
				paid_gems = np.minimum(np.maximum(card_cost - player_cards, 0), player_gems) #正の値
				sum_paid += np.sum(paid_gems)

				if i in buyable_id:
					gain = card[6] #宝石カードが持つ威信ポイント
					buyable_gains.append(gain)
					#buyable_gain += gain
			
			buyable_gain = max(buyable_gains) if len(buyable_gains) > 0 else 0. 
			
			"""
			if buyable_gain > 0:
				print_board(canonical_board)
				print("buyable_gain: ", buyable_gain)
				print()
			"""

			player_score = canonical_board.get_score(player)
			#score_val = 1000. * depth if player_score >= 15. else player_score
			score_val = np.inf if player_score >= 15. else player_score
			val = sum_paid - buyable_gain * 100. - score_val * 10.

			if player == cur_player:
				val_total -= val
			else:
				val_total += val

		return val_total
	

	"""
	#反復深化
	def iterative_deepening(self, node, depth, alpha, beta, deadline, board, player):
		best_action = None
		for depth in range(1, depth + 1):
			value = self.alphabeta(self, node, depth, alpha, beta, deadline, board, player)
			if action is not None:
				best_action = action
			return best_action
	"""

	
	#αβ探索
	def alphabeta(self, node, depth, alpha, beta, deadline, board, player, round):
		is_win = board.check_end_game()
		if depth == 0 or is_win.any() or time.time() >= deadline: #探索停止条件
			#value = self.valueFunc(board, player, depth)
			value = self.valueFuncNN(board.state, player)
			#print("value: ", value)
			return value
		
		is_max_player = player == self.player #現在のノードが自分の番か否か
		next_player = (player+1) % self.num_player

		game_board = Board(self.num_player)
		game_board.copy_state(board.state, True)

		children = node.get_children(self.game, game_board, board.state, player)
		children = self.pruning(children)

		if is_max_player:
			for child in children:
				value = self.alphabeta(child, depth-1, alpha, beta, deadline, child.board, next_player, round)
				alpha = max(alpha, value)
				if beta <= alpha:
					break
			return value
		
		else:
			for child in children:
				value = self.alphabeta(child, depth-1, alpha, beta, deadline, child.board, next_player, round)				
				beta = min(beta, value)
				if beta <= alpha:
					break
			return beta

	# 最低限の枝刈り
	def pruning(self, nodes):
		only_little = False
		is_little = np.array([(node.action > 29 and node.action < 45) or (node.action >= 12+15+3+30 and node.action < 12+15+3+30+20) for node in nodes])
		if is_little.all():
			only_little = True
		
		nodes_pruned = []
		for node in nodes:
			action = node.action
			if not only_little and ((action > 29 and action < 45) or (action >= 12+15+3+30 and action < 12+15+3+30+20)):
				continue

			nodes_pruned.append(node)
		return nodes_pruned
	
	def play(self, board):
		dummy_board = Board(self.num_player)
		dummy_board.copy_state(board, True)

		canonical_board = self.game.getCanonicalForm(board, self.player)
		valids = dummy_board.valid_moves(self.player)
		valids = np.where(valids == True)[0]
		if len(valids) == 1:
			return valids[0]

		start = time.time() #パラメータ
		deadline = start + MAX_SEARCH_TIME
		state_id = self.game.stringRepresentation(self.game.board.state)
		root = Node(state_id, self.player, board, None, 0)
		children = root.get_children(self.game, dummy_board, dummy_board.state, self.player)

		values = {}
		children = self.pruning(children) #枝刈り & ソート
		round = self.game.getRound(board)

		for child in children:
			dummy_board.copy_state(board, True)
			next_player = dummy_board.make_move(child.action, self.player, True)

			#通常のαβ探索
			value = self.alphabeta(
				root, self.depth, -np.inf, np.inf, deadline, dummy_board, next_player, round
			)

			values[child.action] = value
			action_str = self.game.moveToString(child.action, self.player)
			#print("[%02d] expect: %2d, value: %4.2f, -> %s " % (child.action, child.expect, value, action_str))

		best_action, best_value = max(values.items(), key=lambda x: x[1])
		print("best action:", best_action, "best value: ", best_value)

		#if children[0].expect > 3 and (best_value < 1000 or best_value > -1000):
		#	best_action = children[0].action
		
		return best_action