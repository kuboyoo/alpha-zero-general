from GenericNNetWrapper import GenericNNetWrapper







"""
import numpy as np

def judge(scores, num_cards, single_winner, score_max):
		
	if single_winner:
		winners = np.array([1. if s == score_max else -1. for s in scores], dtype=np.float32)
	else: #15点以上で同点のプレイヤーが2人以上いる場合
		num_cards_masked = num_cards.copy()
		winners = np.ones(2, dtype=np.float32) * (-1)
		num_cards_masked[np.where(scores < score_max)] = 999
		num_cards_masked_min = num_cards_masked.min() #15点以上の宝石カード枚数の最小値
		min_ids = np.where(num_cards_masked == num_cards_masked_min)[0]
		winners[min_ids] = 0.01 if len(min_ids) > 1 else 1.

	return winners

def check_end_game(scores):
		score_win = 15
		num_players = len(scores)
		score_max = scores.max()
		end = (score_max >= score_win)
		if not end:
			return np.full(num_players, 0., dtype=np.float32)
		single_winner = ((scores == score_max).sum() == 1)
		#winners = [(1. if single_winner else 0.01) if s == score_max else -1. for s in scores]
		num_cards = np.array([
			14,12
		])
		winners = judge(scores, num_cards, single_winner, score_max)
		return np.array(winners, dtype=np.float32)

scores = np.array([15, 15], dtype=np.int8)
print(check_end_game(scores))
"""
