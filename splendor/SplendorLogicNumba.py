from .SplendorLogic import np_all_nobles, np_all_cards_1, np_all_cards_2, np_all_cards_3, len_all_cards, np_different_gems_up_to_2, np_different_gems_up_to_3, np_2specs_gems_up_to_3, np_cards_symmetries, np_reserve_symmetries
import numpy as np
from numba import njit
import numba

ENABLE_ACTION_RESERVE  = True
ENABLE_ACTION_GIVEBACK = True
NUM_3TAKE_1GIVE = 20
NUM_3TAKE_2GIVE = 30
NUM_2TAKE_DIFF_2GIVE = 60
NUM_2TAKE_SAME_2GIVE = 50
NUM_2TAKE_DIFF_1GIVE = 30
NUM_2TAKE_SAME_1GIVE = 20
NUM_1TAKE_1GIVE = 20
NUM_1TAKEG_1GIVE= 75
NUM_3TAKE_3GIVE = 40
NUM_OF_EXCHANGE = NUM_3TAKE_1GIVE + NUM_3TAKE_2GIVE + NUM_2TAKE_DIFF_2GIVE + NUM_2TAKE_SAME_2GIVE\
								+ NUM_2TAKE_DIFF_1GIVE + NUM_2TAKE_SAME_1GIVE + NUM_1TAKE_1GIVE + NUM_1TAKEG_1GIVE\
							  + NUM_3TAKE_3GIVE

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
mask = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return (32 + 10*num_players + num_players*num_players, 7)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	#return 81
	#return 270 #+exchange
	#return 366 #+rsv exchange +pass
	return 406 #+3-3 exchange

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result

@njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint8), mask[:len(array)])
	return product.sum()

@njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

@njit(cache=True, fastmath=True, nogil=True)
def np_all_axis1(x):
	out = np.ones(x.shape[0], dtype=np.bool8)
	for i in range(x.shape[1]):
		out = np.logical_and(out, x[:, i])
	return out


spec = [
	('num_players'         , numba.int8),
	('current_player_index', numba.int8),
	('num_gems_in_play'    , numba.int8),
	('num_nobles'          , numba.int8),
	('max_moves'           , numba.uint8),
	('score_win'           , numba.int8),

	('state'           , numba.int8[:,:]),
	('bank'            , numba.int8[:,:]),
	('cards_tiers'     , numba.int8[:,:]),
	('nb_deck_tiers'   , numba.int8[:,:]),
	('nobles'          , numba.int8[:,:]),
	('players_gems'    , numba.int8[:,:]),
	('players_nobles'  , numba.int8[:,:]),
	('players_cards'   , numba.int8[:,:]),
	('players_reserved', numba.int8[:,:]),
	('give_ids'        , numba.int8[:,:,:]),
	('give_ids3'       , numba.int8[:,:])
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		n = num_players
		self.num_players = n
		self.current_player_index = 0
		self.num_gems_in_play = {2: 4, 3: 5, 4: 7}[n]
		self.num_nobles = {2:3, 3:4, 4:5}[n]
		self.max_moves = 62 * num_players
		self.score_win = 15
		self.state = np.zeros(observation_size(self.num_players), dtype=np.int8)
		self.give_ids = np.array([
			[[3,4,0,0,0,0,0,0,0,0], 
			[2,4,0,0,0,0,0,0,0,0], 
			[2,3,0,0,0,0,0,0,0,0], 
			[1,4,0,0,0,0,0,0,0,0], 
			[1,3,0,0,0,0,0,0,0,0], 
			[1,2,0,0,0,0,0,0,0,0], 
			[0,4,0,0,0,0,0,0,0,0], 
			[0,3,0,0,0,0,0,0,0,0], 
			[0,2,0,0,0,0,0,0,0,0], 
			[0,1,0,0,0,0,0,0,0,0]],

			[[14,18,19,0,0,0,0,0,0,0], 
			[13,17,19,0,0,0,0,0,0,0], 
			[12,17,18,0,0,0,0,0,0,0], 
			[11,16,19,0,0,0,0,0,0,0], 
			[10,16,18,0,0,0,0,0,0,0], 
			[9,16,17,0,0,0,0,0,0,0], 
			[8,15,19,0,0,0,0,0,0,0], 
			[7,15,18,0,0,0,0,0,0,0], 
			[6,15,17,0,0,0,0,0,0,0], 
			[5,15,16,0,0,0,0,0,0,0]],

			[[12,13,14,17,18,19,0,0,0,0],
			[10,11,14,16,18,19,0,0,0,0], 
			[9,11,13,17,16,19,0,0,0,0],
			[9,10,12,17,16,18,0,0,0,0],
			[7,8,14,15,19,18,0,0,0,0],
			[6,8,13,15,19,17,0,0,0,0],
			[6,7,12,15,18,17,0,0,0,0],
			[5,8,11,15,19,16,0,0,0,0],
			[5,7,10,15,18,16,0,0,0,0],
			[6,5,9,15,16,17,0,0,0,0]],
			
			[[9,12,13,10,11,14,17,16,18,19],
			[6,7,8,12,13,14,15,17,18,19],
			[5,7,8,10,11,14,15,16,18,19], 
			[6,5,8,9,13,11,15,17,16,19],
			[6,5,7,9,12,10,15,17,16,18],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0]],

			[[2,3,4,0,0,0,0,0,0,0],
			[1,3,4,0,0,0,0,0,0,0],
			[1,2,4,0,0,0,0,0,0,0],
			[1,2,3,0,0,0,0,0,0,0],
			[0,3,4,0,0,0,0,0,0,0],
			[0,2,4,0,0,0,0,0,0,0],
			[0,2,3,0,0,0,0,0,0,0],
			[0,1,4,0,0,0,0,0,0,0],
			[0,1,3,0,0,0,0,0,0,0],
			[0,1,2,0,0,0,0,0,0,0]],

			[[1,2,3,4,0,0,0,0,0,0],
			[0,2,3,4,0,0,0,0,0,0],
			[0,1,3,4,0,0,0,0,0,0],
			[0,1,2,4,0,0,0,0,0,0],
			[0,1,2,3,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0]]
		], dtype=np.int8)

		#[take3, give2, give2]
		self.give_ids3 = np.array(
			[[0, 3, 18], 
    	 [0, 18, 4], 
			 [0, 3, 19],
			 [0, 19, 4],
			 [1, 2, 17],
			 [1, 17, 4],
			 [1, 2, 19],
			 [1, 19, 4],
			 [2, 2, 17],
			 [2, 17, 3],
			 [2, 2, 18],
			 [2, 18, 3],
			 [3, 1, 16],
			 [3, 16, 4],
			 [3, 1, 19],
			 [3, 19, 4],
			 [4, 1, 16],
			 [4, 16, 3],
			 [4, 1, 18],
			 [4, 18, 3],
			 [5, 1, 16],
			 [5, 16, 2],
			 [5, 1, 17],
			 [5, 17, 2],
			 [6, 0, 15],
			 [6, 15, 4],
			 [6, 0, 19],
			 [6, 19, 4],
			 [7, 0, 15],
			 [7, 15, 3],
			 [7, 0, 18],
			 [7, 18, 3],
			 [8, 0, 15],
			 [8, 15, 2],
			 [8, 0, 17],
			 [8, 17, 2],
			 [9, 0, 15],
			 [9, 15, 1],
			 [9, 0, 16],
			 [9, 16, 1]], dtype=np.int8
		)

		self.init_game()

	def get_score(self, player):
		card_points  = self.players_cards[player, idx_points]
		noble_points = self.players_nobles[player*3:player*3+3, idx_points].sum()
		return card_points + noble_points

	def init_game(self):
		self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

		# Bank
		self.bank[:] = np.array([[self.num_gems_in_play]*5 + [5, 0]], dtype=np.int8)
		# Decks
		for tier in range(3):
			nb_deck_cards_per_color = len_all_cards[tier]
			# HOW MANY cards per color are in deck of tier 0, pratical for NN
			self.nb_deck_tiers[2*tier,:5] = nb_deck_cards_per_color
			# WHICH cards per color are in deck of tier 0, pratical for logic
			self.nb_deck_tiers[2*tier+1,:5] = my_packbits(np.ones(nb_deck_cards_per_color, dtype=np.int8))
		# Tiers
		for tier in range(3):
			for index in range(4):
				self._fill_new_card(tier, index, False)
		# Nobles
		nobles_indexes = np.random.choice(len(np_all_nobles), size=self.num_nobles, replace=False)
		for i, index in enumerate(nobles_indexes):
			self.nobles[i, :] = np_all_nobles[index]

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		result = np.zeros(action_size(), dtype=np.bool_)
		result[0         :12]            = self._valid_buy(player)
		result[12        :12+15]         = self._valid_reserve(player)
		result[12+15     :12+15+3]       = self._valid_buy_reserve(player)
		result[12+15+3   :12+15+3+30]    = np.concatenate((self._valid_get_gems(player) , self._valid_get_gems_identical(player)))
		get_flgs = np.concatenate((self._valid_get_gems(player, False) , self._valid_get_gems_identical(player, False)))
		giv_flgs = np.concatenate((self._valid_give_gems(player), self._valid_give_gems_identical(player)))
		giv_flgs3= self._valid_give_gems3(player)
		rsv_flg  = self._valid_reserve(player, False)
		result[12+15+3+30:-1] = self._valid_exchange(player, get_flgs, giv_flgs, giv_flgs3, rsv_flg)
		result[-1] = True if not np.any(result[:-1]) else False  #何も行動できない場合のみpass可

		return result

	def make_move(self, move, player, deterministic):
		if   move < 12:
			self._buy(move, player, deterministic)
		elif move < 12+15:
			self._reserve(move-12, player, deterministic)
		elif move < 12+15+3:
			self._buy_reserve(move-12-15, player)
		elif move < 12+15+3+30: #[30-59]までがtake only 
			# 30-34 (5) : +1 
			# 35-44 (10): +2(異)
			# 45-54 (10): +3
			# 55-59 (5) : +2(同)
			self._get_gems(move-12-15-3, player)
		elif move < 12+15+3+30 +210+20: #60-289
			self._give_and_get_gems(move-60, player)
		elif move < 12+15+3+30 +210+20 + 75: #290-364
			self._reserve_and_give(move-60-210-20, player, deterministic)
		else: #365-404
			self._give_and_get_gems(move-60, player)

		self.bank[0][idx_points] += 1 # Count number of rounds

		return (player+1)%self.num_players

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.bank             = self.state[0         :1          ,:]	# 1
		self.cards_tiers      = self.state[1         :25         ,:]	# 2*12
		self.nb_deck_tiers    = self.state[25        :31         ,:]	# 6
		self.nobles           = self.state[31        :32+n       ,:]	# N+1
		self.players_gems     = self.state[32+n      :32+2*n     ,:]	# N
		self.players_nobles   = self.state[32+2*n    :32+3*n+n*n ,:]	# N*(N+1)
		self.players_cards    = self.state[32+3*n+n*n:32+4*n+n*n ,:]	# N
		self.players_reserved = self.state[32+4*n+n*n:32+10*n+n*n,:]	# 6*N
	
	#勝敗引き分け判定
	def judge(self, scores, num_cards, single_winner, score_max):
		
		if single_winner:
			winners = np.array([1. if s == score_max else -1. for s in scores], dtype=np.float32)
		else: #15点以上で同点のプレイヤーが2人以上いる場合
			num_cards_masked = num_cards.copy()
			winners = np.ones(self.num_players, dtype=np.float32) * (-1)
			num_cards_masked[np.where(scores < score_max)] = 999
			num_cards_masked_min = num_cards_masked.min() #15点以上の宝石カード枚数の最小値
			min_ids = np.where(num_cards_masked == num_cards_masked_min)[0]
			winners[min_ids] = 0.01 if len(min_ids) > 1 else 1.

		return winners

	def check_end_game(self):

		if self.get_round() % self.num_players != 0: # Check only when 1st player is about to play
			return np.full(self.num_players, 0., dtype=np.float32)
		
		scores = np.array([self.get_score(p) for p in range(self.num_players)], dtype=np.int8)
		score_max = scores.max()
		end = (score_max >= self.score_win) or (self.get_round() >= self.max_moves)
		if not end:
			return np.full(self.num_players, 0., dtype=np.float32)
		single_winner = ((scores == score_max).sum() == 1)
		num_cards = np.array([self.players_cards[p][:5].sum() for p in range(self.num_players)], dtype=np.int8)
		winners = self.judge(scores, num_cards, single_winner, score_max)
		#winners = [(1. if single_winner else 0.01) if s == score_max else -1. for s in scores]
		return winners

	# if n=1, transform P0 to Pn, P1 to P0, ... and Pn to Pn-1
	# else do this action n times
	def swap_players(self, nb_swaps):
		def _roll_in_place_axis0(array, shift):
			tmp_copy = array.copy()
			size0 = array.shape[0]
			for i in range(size0):
				array[i,:] = tmp_copy[(i+shift)%size0,:]
		_roll_in_place_axis0(self.players_gems    , 1*nb_swaps)
		_roll_in_place_axis0(self.players_nobles  , 3*nb_swaps)
		_roll_in_place_axis0(self.players_cards   , 1*nb_swaps)
		_roll_in_place_axis0(self.players_reserved, 6*nb_swaps)

	def get_symmetries(self, policy, valid_actions):
		def _swap_cards(cards, permutation):
			full_permutation = [2*p+i for p in permutation for i in range(2)]
			cards_copy = cards.copy()
			for i in range(len(permutation)*2):
				cards[i, :] = cards_copy[full_permutation[i], :]
		def _copy_and_permute(array, permutation, start_index):
			new_array = array.copy()
			for i, p in enumerate(permutation):
				new_array[start_index+i] = array[start_index+p]
			return new_array
		def _copy_and_permute2(array, permutation, start_index, other_start_index):
			new_array = array.copy()
			for i, p in enumerate(permutation):
				new_array[start_index      +i] = array[start_index      +p]
				new_array[other_start_index+i] = array[other_start_index+p]
			return new_array

		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		# Permute common cards within same tier
		for tier in range(3):
			for permutation in np_cards_symmetries:
				cards_tiers_backup = self.cards_tiers.copy()
				_swap_cards(self.cards_tiers[8*tier:8*tier+8, :], permutation)
				new_policy = _copy_and_permute2(policy, permutation, 4*tier, 12+4*tier)
				new_valid_actions = _copy_and_permute2(valid_actions, permutation, 4*tier, 12+4*tier)
				symmetries.append((self.state.copy(), new_policy, new_valid_actions))
				self.cards_tiers[:] = cards_tiers_backup
		
		# Permute reserved cards
		for player in range(self.num_players):
			nb_reserved_cards = self._nb_of_reserved_cards(player)
			for permutation in np_reserve_symmetries[nb_reserved_cards]:
				if permutation[0] < 0:
					continue
				players_reserved_backup = self.players_reserved.copy()
				_swap_cards(self.players_reserved[6*player:6*player+6, :], permutation)
				if player == 0:
					new_policy = _copy_and_permute(policy, permutation, 12+15)
					new_valid_actions = _copy_and_permute(valid_actions, permutation, 12+15)
				else:
					new_policy = policy.copy()
					new_valid_actions = valid_actions.copy()
				symmetries.append((self.state.copy(), new_policy, new_valid_actions))
				self.players_reserved[:] = players_reserved_backup

		return symmetries

	def get_round(self):
		return self.bank[0].astype(np.uint8)[idx_points]

	def _get_deck_card(self, tier):
		nb_remaining_cards_per_color = self.nb_deck_tiers[2*tier,:5]
		if nb_remaining_cards_per_color.sum() == 0: # no more cards
			return None
		
		# First we chose color randomly, then we pick a card 
		color = my_random_choice(nb_remaining_cards_per_color/nb_remaining_cards_per_color.sum())
		remaining_cards = my_unpackbits(self.nb_deck_tiers[2*tier+1, color])
		card_index = my_random_choice(remaining_cards/remaining_cards.sum())
		# Update internals
		remaining_cards[card_index] = 0
		self.nb_deck_tiers[2*tier+1, color] = my_packbits(remaining_cards)
		self.nb_deck_tiers[2*tier, color] -= 1

		if tier == 0:
			card = np_all_cards_1[color][card_index]
		elif tier == 1:
			card = np_all_cards_2[color][card_index]
		else:
			card = np_all_cards_3[color][card_index]
		return card

	def _fill_new_card(self, tier, index, deterministic):
		self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = 0
		if not deterministic:
			card = self._get_deck_card(tier)
			if card is not None:
				self.cards_tiers[8*tier+2*index:8*tier+2*index+2] = card

	def _buy_card(self, card0, card1, player):
		card_cost = card0[:5]
		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(card_cost - player_gems - player_cards, 0).sum()
		# Apply changes
		paid_gems = np.minimum(np.maximum(card_cost - player_cards, 0), player_gems)
		player_gems -= paid_gems
		self.bank[0][:5] += paid_gems
		self.players_gems[player][idx_gold] -= missing_colors
		self.bank[0][idx_gold] += missing_colors
		self.players_cards[player] += card1

		self._give_nobles_if_earned(player)

	def _valid_buy(self, player):
		cards_cost = self.cards_tiers[:2*12:2,:5]

		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy(self, i, player, deterministic):
		tier, index = divmod(i, 4)
		self._buy_card(self.cards_tiers[2*i], self.cards_tiers[2*i+1], player)
		self._fill_new_card(tier, index, deterministic)

	def _valid_reserve(self, player, is_limit=True):
		if ((not ENABLE_ACTION_RESERVE) or (self.players_gems[player].sum() == 10 and self.bank[0][idx_gold] > 0)) and is_limit:
			return np.zeros(12+3, dtype=np.int8)
		not_empty_cards = np.vstack((self.cards_tiers[:2*12:2,:5], self.nb_deck_tiers[::2, :5])).sum(axis=1) != 0

		allowed_reserved_cards = 3
		empty_slot = (self.players_reserved[6*player+2*(allowed_reserved_cards-1)+1][:5].sum() == 0)
		return np.logical_and(not_empty_cards, empty_slot).astype(np.int8)

	def _reserve(self, i, player, deterministic):
		# Detect empty reserve slot
		reserve_slots = [6*player+2*i for i in range(3)]
		for slot in reserve_slots:
			if self.players_reserved[slot,:5].sum() == 0:
				empty_slot = slot
				break
		
		if i < 12: # reserve visible card
			tier, index = divmod(i, 4)
			self.players_reserved[empty_slot:empty_slot+2] = self.cards_tiers[8*tier+2*index:8*tier+2*index+2]
			self._fill_new_card(tier, index, deterministic)
		else:      # reserve from deck
			if not deterministic:
				tier = i - 12
				self.players_reserved[empty_slot:empty_slot+2] = self._get_deck_card(tier)
		
		if self.bank[0][idx_gold] > 0:# and self.players_gems[player].sum() <= 9:
			self.players_gems[player][idx_gold] += 1
			self.bank[0][idx_gold] -= 1

	def _valid_buy_reserve(self, player):
		card_index = np.arange(3)
		cards_cost = self.players_reserved[6*player+2*card_index,:5]

		player_gems = self.players_gems[player][:5]
		player_cards = self.players_cards[player][:5]
		missing_colors = np.maximum(cards_cost - player_gems - player_cards, 0).sum(axis=1)
		enough_gems_and_gold = missing_colors <= self.players_gems[player][idx_gold]
		not_empty_cards = cards_cost.sum(axis=1) != 0

		return np.logical_and(enough_gems_and_gold, not_empty_cards).astype(np.int8)

	def _buy_reserve(self, i, player):
		start_index = 6*player+2*i
		self._buy_card(self.players_reserved[start_index], self.players_reserved[start_index+1], player)
		# shift remaining reserve to the beginning
		if i < 2:
			self.players_reserved[start_index:6*player+4] = self.players_reserved[start_index+2:6*player+6]
		self.players_reserved[6*player+4:6*player+6] = 0 # empty last reserve slot

	def _valid_get_gems(self, player, is_limit=True):
		gems = np_different_gems_up_to_3[:,:5]
		enough_in_bank = np_all_axis1((self.bank[0][:5] - gems) >= 0)
		num_player_gems = self.players_gems[player].sum()
		num_spec_bank_gems = np.count_nonzero(self.bank[0][:5])
		not_too_many_gems = num_player_gems + gems.sum(axis=1) <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8) if is_limit else enough_in_bank.astype(np.int8)

		#1 or 異色2枚取りできるのは、銀行とプレイヤーのトークン枚数が条件を満たす場合のみ
		if num_player_gems != 9 and num_spec_bank_gems != 1 and is_limit:
			result[:5] = False
		if num_player_gems != 8 and num_spec_bank_gems != 2 and is_limit:
			result[5:15] = False

		return result

	def _valid_get_gems_identical(self, player, is_limit=True):
		colors = np.arange(5)
		enough_in_bank = self.bank[0][colors] >= 4
		not_too_many_gems = self.players_gems[player].sum() + 2 <= 10
		result = np.logical_and(enough_in_bank, not_too_many_gems).astype(np.int8) if is_limit else enough_in_bank.astype(np.int8)
		return result

	def _get_gems(self, i, player):
		if i < np_different_gems_up_to_3.shape[0]: # Different gems
			gems = np_different_gems_up_to_3[i][:5]
		else:                                      # 2 identical gems
			color = i - np_different_gems_up_to_3.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		self.bank[0][:5] -= gems
		self.players_gems[player][:5] += gems

	def _valid_give_gems(self, player):
		if not ENABLE_ACTION_GIVEBACK:
			return np.zeros(np_different_gems_up_to_2.shape[0], dtype=np.int8)
		gems = np_different_gems_up_to_2[:,:5]
		result = np_all_axis1((self.players_gems[player][:5] - gems) >= 0).astype(np.int8)
		return result
	
	def _valid_give_gems3(self, player):
		if not ENABLE_ACTION_GIVEBACK:
			return np.zeros(np_2specs_gems_up_to_3.shape[0], dtype=np.int8)
		gems = np_2specs_gems_up_to_3[:,:5]
		result = np_all_axis1((self.players_gems[player][:5] - gems) >= 0).astype(np.int8)
		return result

	def _valid_give_gems_identical(self, player):
		if not ENABLE_ACTION_GIVEBACK:
			return np.zeros(5, dtype=np.int8)
		colors = np.arange(5)
		return (self.players_gems[player][colors] >= 2).astype(np.int8)

	def _valid_exchange(self, player, get_flgs, giv_flgs, giv_flgs3, rsv_flg):
		num_of_tokens = self.players_gems[player].sum()
		if num_of_tokens <= 7:
			return np.zeros(NUM_OF_EXCHANGE, dtype=np.int8)

		same2, dif2, dif3 = get_flgs[25:], get_flgs[5:15], get_flgs[15:25]

		take3_give1  = np.zeros(NUM_3TAKE_1GIVE, dtype=np.int8)
		take3_give2  = np.zeros(NUM_3TAKE_2GIVE, dtype=np.int8)
		take3_give3  = np.zeros(NUM_3TAKE_3GIVE, dtype=np.int8)
		take2d_give2 = np.zeros(NUM_2TAKE_DIFF_2GIVE, dtype=np.int8)
		take2s_give2 = np.zeros(NUM_2TAKE_SAME_2GIVE, dtype=np.int8)
		take2d_give1 = np.zeros(NUM_2TAKE_DIFF_1GIVE, dtype=np.int8)
		take2s_give1 = np.zeros(NUM_2TAKE_SAME_1GIVE, dtype=np.int8)
		take1g_give1 = np.zeros(NUM_1TAKEG_1GIVE, dtype=np.int8)
		take1_give1  = np.zeros(NUM_1TAKE_1GIVE, dtype=np.int8)

		if num_of_tokens == 8:
			#3take 1give
			take3_flgs_for_give1  = np.repeat(dif3, 2)
			give1_flgs_for_take3  = giv_flgs[self.give_ids[0, :, 0:2].flatten()]
			take3_give1 = np.logical_and(take3_flgs_for_give1, give1_flgs_for_take3).astype(np.int8)

		elif num_of_tokens == 9:
			#3take 2give
			take3_flgs_for_give2 = np.repeat(dif3, 3)
			give2_flgs_for_take3 = giv_flgs[self.give_ids[1, :, 0:3].flatten()]
			take3_give2 = np.logical_and(take3_flgs_for_give2, give2_flgs_for_take3).astype(np.int8)
		
			#2take 1give
			take2d_flgs_for_give1 = np.repeat(dif2, 3)
			give1_flgs_for_take2d = giv_flgs[self.give_ids[4, :, 0:3].flatten()]
			take2d_give1 = np.logical_and(take2d_flgs_for_give1, give1_flgs_for_take2d).astype(np.int8)

			#2take 1give
			take2s_flgs_for_give1 = np.repeat(same2, 4)
			give1_flgs_for_take2s = giv_flgs[self.give_ids[5, 0:5, 0:4].flatten()]
			take2s_give1 = np.multiply(take2s_flgs_for_give1, give1_flgs_for_take2s).astype(np.int8)
		
		else: #10tokens
			#2take 2give
			take2d_flgs_for_give2 = np.repeat(dif2, 6)
			give2_flgs_for_take2d = giv_flgs[self.give_ids[2, :, 0:6].flatten()]
			take2d_give2 = np.logical_and(take2d_flgs_for_give2, give2_flgs_for_take2d).astype(np.int8)

			#2take 2give
			take2s_flgs_for_give2 = np.repeat(same2, 10)
			give2_flgs_for_take2s = giv_flgs[self.give_ids[3, 0:5, :].flatten()]
			take2s_give2 = np.logical_and(take2s_flgs_for_give2, give2_flgs_for_take2s).astype(np.int8)

			#1take(not gold) 1give
			take1_flgs = np.repeat(get_flgs[:5], 4)
			give_ids_for_take1 = np.array([1,2,3,4, 0,2,3,4, 0,1,3,4, 0,1,2,4, 0,1,2,3])
			give1_flgs_for_take1 = giv_flgs[give_ids_for_take1]
			take1_give1 = np.logical_and(take1_flgs, give1_flgs_for_take1).astype(np.int8)

			#3take 3give
			take3_give3 = np.logical_and(np.repeat(dif3, 4), giv_flgs3).astype(np.int8)

			if self.bank[0][idx_gold] > 0:
				#1take(gold) 1give
				rsv_flgs = np.repeat(rsv_flg, 5)
				give1_flgs_for_take1g = np.repeat(giv_flgs[:5], 15).reshape(-1, 15).T.flatten()
				take1g_give1 = np.logical_and(rsv_flgs, give1_flgs_for_take1g).astype(np.int8)

		return np.concatenate((take3_give1, take3_give2, take2d_give2, take2s_give2, take2d_give1, take2s_give1, take1_give1, take1g_give1, take3_give3))

	def _give_gems(self, i, player):
		if i < np_different_gems_up_to_2.shape[0]: # Different gems (0-14)
			gems = np_different_gems_up_to_2[i][:5]
		else: # 2 identical gems (15-19)
			color = i - np_different_gems_up_to_2.shape[0]
			gems = np.zeros(5, dtype=np.int8)
			gems[color] = 2
		
		self.bank[0][:5] += gems
		self.players_gems[player][:5] -= gems

	#トークンの交換
	def _give_and_get_gems(self, i, player):
		if i < 20: #+3 -1 (20) id: 60-79
			give_i = self.give_ids[0]
			id_3_get = i // 2
			id_1_give = give_i[id_3_get][i % 2]
			self._get_gems(id_3_get+15, player)
			self._give_gems(id_1_give, player)
		
		elif i < 20+30: #+3 -2 (30) id: 80-109
			i -= 20
			give_i = self.give_ids[1]
			id_3_get = i // 3
			id_2_give = give_i[id_3_get][i % 3]
			self._get_gems(id_3_get+15, player)
			self._give_gems(id_2_give, player)
		
		elif i < 20+30+60: #+2(異) -2 (60) id: 110-169
			i -= (20+30)
			give_i = self.give_ids[2]
			id_2_get = i // 6
			id_2_give = give_i[id_2_get][i % 6]
			self._get_gems(id_2_get+5, player)
			self._give_gems(id_2_give, player)
		
		elif i < 20+30+60+50: #+2(同) -2 (50) id: 170-219
			i -= (20+30+60)
			give_i = self.give_ids[3]
			id_2_get = i // 10
			id_2_give = give_i[id_2_get][i % 10]
			self._get_gems(id_2_get+25, player)
			self._give_gems(id_2_give, player)
		
		elif i < 20+30+60+50+30: #+2(異) -1 (30) id: 220-249
			i -= (20+30+60+50)
			give_i = self.give_ids[4]
			id_2_get = i // 3
			id_1_give = give_i[id_2_get][i % 3]
			self._get_gems(id_2_get+5, player)
			self._give_gems(id_1_give, player)

		elif i < 20+30+60+50+30+20: #+2(同) -1 (20) id: 250-269
			i -= (20+30+60+50+30)
			give_i = self.give_ids[5]
			id_2_get = i // 4
			id_1_give = give_i[id_2_get][i % 4]
			self._get_gems(id_2_get+25, player)
			self._give_gems(id_1_give, player)
		
		elif i < 20+30+60+50+30+20+20: #+1(金以外) -1 (20) id: 270-289
			i -= (20+30+60+50+30+20)
			_give_ids = np.array([1,2,3,4, 0,2,3,4, 0,1,3,4, 0,1,2,4, 0,1,2,3])
			self._get_gems(i // 4, player)
			self._give_gems(_give_ids[i], player)
		
		else: #+3 -3(金以外) id: 365-404
			i -= 305
			take_id, give_id1, give_id2 = self.give_ids3[i]
			self._get_gems(take_id+15, player)
			self._give_gems(give_id1, player)
			self._give_gems(give_id2, player)

	#予約+トークンの返却(+金 -1の交換)
	def _reserve_and_give(self, i, player, determistic):
		self._reserve(i // 5, player, determistic)
		self._give_gems(i % 5, player)

	def _give_nobles_if_earned(self, player):
		for i_noble in range(self.num_nobles):
			noble = self.nobles[i_noble][:5]
			if noble.sum() > 0 and np.all(self.players_cards[player][:5] >= noble):
				self.players_nobles[self.num_nobles*player+i_noble] = self.nobles[i_noble]
				self.nobles[i_noble] = 0

	def _nb_of_reserved_cards(self, player):
		for card in range(3):
			if self.players_reserved[6*player+2*card,:5].sum() == 0:
				return card # slot 'card' is empty, there are 'card' cards
		return 3
