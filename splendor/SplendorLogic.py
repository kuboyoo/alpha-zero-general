import numpy as np
from colorama import Style, Fore, Back
import random
import itertools

give_ids = [
	np.array([[3,4], [2,4], [2,3], [1,4], [1,3], [1,2], [0,4], [0,3], [0,2], [0,1]]),
	np.array([[14,18,19], [13,17,19], [12,17,18], [11,16,19], [10,16,18], [9,16,17], [8,15,19], [7,15,18], [6,15,17], [5,15,16]]),
	np.array([[12,13,14,17,18,19], [10,11,14,16,18,19], [9,11,13,17,16,19], [9,10,12,17,16,18], [7,8,14,15,19,18], [6,8,13,15,19,17], [6,7,12,15,18,17], [5,8,11,15,19,16], [5,7,10,15,18,16], [6,5,9,15,16,17]]),
	np.array([[9,12,13,10,11,14,17,16,18,19], [6,7,8,12,13,14,15,17,18,19], [5,7,8,10,11,14,15,16,18,19],  [6,5,8,9,13,11,15,17,16,19], [6,5,7,9,12,10,15,17,16,18]]),
	np.array([[2,3,4], [1,3,4], [1,2,4], [1,2,3], [0,3,4], [0,2,4], [0,2,3], [0,1,4], [0,1,3], [0,1,2]]),
	np.array([[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]),
	np.array([1,2,3,4, 0,2,3,4, 0,1,3,4, 0,1,2,4, 0,1,2,3])
]

give_ids3 = np.array(
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

def move_to_str(move, short=False):
	color_names = ['white', 'blue', 'green', 'red', 'black', 'gold']
	if   move < 12:
		tier, index = divmod(move, 4)
		return f'buy tier{tier}-card{index}' if short else f'buy from tier {tier} index {index}'
	elif move < 12+15:
		if move < 12+12:
			tier, index = divmod(move-12, 4)
			return f'rsv t{tier}-c{index}' if short else f'reserve from tier {tier} index {index}'
		else:
			tier = move-12-12
			return f'rsv t{tier}-deck' if short else f'reserve from deck of tier {tier}'
	elif move < 12+15+3:
		index = move-12-15
		return f'buy rsv{index}'if short else f'buy from reserve {index}'
	elif move < 12+15+3+30:
		i = move - 12-15-3
		if i < len(list_different_gems_up_to_3):
			if short:
				gems_str = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
				return f'{" ".join(gems_str)}'
			else:
				gems_str = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
				return f'take {", ".join(gems_str)}'		
		else:
			if short:
				return f'{light_colors[i-len(list_different_gems_up_to_3)] + "    " + Style.RESET_ALL}'
			else:
				return f'take 2 gems of color {color_names[i-len(list_different_gems_up_to_3)]}'
	
	elif move < 12+15+3+30 +20: #i)
		id = move - (12+15+3+30)
		i = id // 2
		j = give_ids[0][i][id % 2]
		if short:
			gems_str_take = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i+15][:5]) if v != 0]
			gems_str_give = [ light_colors[j] + "  " + Style.RESET_ALL for j, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'{" ".join(gems_str_take)} - {" ".join(gems_str_give)}'
		else:
			gems_str_take = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i+15][:5]) if v != 0]
			gems_str_give = [str(v)+" "+color_names[j] for j, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'take {", ".join(gems_str_take)} and give back {", ".join(gems_str_give)}'
	
	elif move < 12+15+3+30 +20+30: #ii)
		id = move - (12+15+3+30 +20)
		i = id // 3
		j = give_ids[1][i][id % 3]
		if short:
			gems_str_take = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i+15][:5]) if v != 0]
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'{light_colors[j-len(list_different_gems_up_to_2)] + "    " + Style.RESET_ALL}'
			else:
				gems_str_give = [ light_colors[jj] + "  " + Style.RESET_ALL for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = " ".join(gems_str_give)
			return f'{" ".join(gems_str_take)} - {gems_str_give}'
		else:
			gems_str_take = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i+15][:5]) if v != 0]
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'2 {color_names[j-len(list_different_gems_up_to_2)]}'
			else:
				gems_str_give = [str(v)+" "+color_names[jj] for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = ", ".join(gems_str_give)
			return f'take {", ".join(gems_str_take)} and give back {gems_str_give}'
		
	elif move < 12+15+3+30 +20+30+60: #iii)
		id = move - (12+15+3+30 +20+30)
		i = id // 6
		j = give_ids[2][i][id % 6]
		if short:
			gems_str_take = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i+5][:5]) if v != 0]
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'{light_colors[j-len(list_different_gems_up_to_2)] + "    " + Style.RESET_ALL}'
			else:
				gems_str_give = [ light_colors[jj] + "  " + Style.RESET_ALL for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = " ".join(gems_str_give)
			return f'{" ".join(gems_str_take)} - {gems_str_give}'
		else:
			gems_str_take = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i+5][:5]) if v != 0]
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'2 {color_names[j-len(list_different_gems_up_to_2)]}'
			else:
				gems_str_give = [str(v)+" "+color_names[jj] for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = ", ".join(gems_str_give)
			return f'take {", ".join(gems_str_take)} and give back {gems_str_give}'
	
	elif move < 12+15+3+30 +20+30+60+50: #iv)
		id = move - (12+15+3+30 +20+30+60)
		i = id // 10
		j = give_ids[3][i][id % 10]
		if short:
			gems_str_take = light_colors[i] + "  " + Style.RESET_ALL
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'{light_colors[j-len(list_different_gems_up_to_2)] + "    " + Style.RESET_ALL}'
			else:
				gems_str_give = [ light_colors[jj] + "  " + Style.RESET_ALL for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = " ".join(gems_str_give)
			return f'{gems_str_take}{gems_str_take} - {gems_str_give}'
		else:
			if j >= len(np_different_gems_up_to_2):
				gems_str_give = f'2 {color_names[j-len(list_different_gems_up_to_2)]}'
			else:
				gems_str_give = [str(v)+" "+color_names[jj] for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
				gems_str_give = ", ".join(gems_str_give)
			return f'take 2 {color_names[i]} and give back {gems_str_give}'
			
	elif move < 12+15+3+30 +20+30+60+50+30: #v)
		id = move - (12+15+3+30 +20+30+60+50)
		i = id // 3
		j = give_ids[4][i][id % 3]
		if short:
			gems_str_take = [ light_colors[i] + "  " + Style.RESET_ALL  for i, v in enumerate(list_different_gems_up_to_3[i+5][:5]) if v != 0]
			gems_str_give = [ light_colors[jj] + "  " + Style.RESET_ALL for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'{" ".join(gems_str_take)} - {" ".join(gems_str_give)}'
		else:
			gems_str_take = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i+5][:5]) if v != 0]
			gems_str_give = [str(v)+" "+color_names[jj] for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'take {", ".join(gems_str_take)} and give back {", ".join(gems_str_give)}'
	
	elif move < 12+15+3+30 +20+30+60+50+30+20: #vi)
		id = move - (12+15+3+30 +20+30+60+50+30)
		i = id // 4
		j = give_ids[5][i][id % 4]
		if short:
			gems_str_take = light_colors[i] + "  " + Style.RESET_ALL
			gems_str_give = [ light_colors[jj] + "  " + Style.RESET_ALL for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'{gems_str_take}{gems_str_take} - {" ".join(gems_str_give)}'
		else:
			gems_str_give = [str(v)+" "+color_names[jj] for jj, v in enumerate(list_different_gems_up_to_2[j][:5]) if v != 0]
			return f'take 2 {color_names[i]}and give back {", ".join(gems_str_give)}'

	elif move < 12+15+3+30 +20+30+60+50+30+20+20: #vii)
		id = move - (12+15+3+30 +20+30+60+50+30+20)
		i = id // 4
		j = give_ids[6][id]
		gems_str_take = light_colors[i] + "  " + Style.RESET_ALL if short else f'take {color_names[i]}'
		gems_str_give = "- " + light_colors[j] + "  " + Style.RESET_ALL if short else f'and give back {color_names[j]}'
		return f'{gems_str_take} {gems_str_give}'

	elif move < 12+15+3+30 +20+30+60+50+30+20+20+75: #viii)
		id = move - (12+15+3+30 +20+30+60+50+30+20+20)
		i = id // 5
		j = id % 5

		gems_str_give = light_colors[j] + "  " + Style.RESET_ALL if short else f'give back {color_names[j]}'

		if i < 12:
			tier, index = divmod(i, 4)
			return f'rsv t{tier}-c{index} - {gems_str_give}' if short else f'reserve from tier {tier} index {index} and {gems_str_give}'
		else:
			tier = i-12
			return f'rsv t{tier}-deck - {gems_str_give}' if short else f'reserve from deck of tier {tier} and {gems_str_give}'
	
	elif move < 12+15+3+30 +20+30+60+50+30+20+20+75 +40: #xi)
		id = move - (12+15+3+30 +20+30+60+50+30+20+20+75)
		i = id // 4 + 15 #take 3(異)
		if short:
			gems_str_take = [ light_colors[i] + "  " + Style.RESET_ALL for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0 for _ in range(v)]
			gems_str_give = [ light_colors[j] + "  " + Style.RESET_ALL for j, v in enumerate(list_2specs_gems_up_to3[id][:5]) if v != 0 for _ in range(v)]
			return f'{" ".join(gems_str_take)} - {" ".join(gems_str_give)}'
		else:
			gems_str_take = [str(v)+" "+color_names[i] for i, v in enumerate(list_different_gems_up_to_3[i][:5]) if v != 0]
			gems_str_give = [str(v)+" "+color_names[j] for j, v in enumerate(list_2specs_gems_up_to3[id][:5]) if v != 0]
			return f'take {", ".join(gems_str_take)} and give back {", ".join(gems_str_give)}'
	else:
		return f'nothing' if short else f'do nothing'
	

def row_to_str(row, n=2):
	if row < 1:
		return 'bank'
	if row < 25:
		tier, index = divmod(row-1, 4)
		cost_or_value = ((row-1)%2 == 0)
		return f'Card in tier {tier} index {index//2} ' + ('cost' if cost_or_value else 'value')
	if row < 28:
		return f'Nb cards in deck of tier {row-25}'
	if row < 29+n:
		return f'Nobles num {row-28}'
	if row < 29+2*n:
		return f'Nb of gems of player {row-29-n}/{n}'
	if row < 29+5*n:
		player, index = divmod(row-29-2*n, 3)
		return f'Noble {index} earned by player {player}/{n}'
	if row < 29+6*n:
		return f'Cards of player {row-29-5*n}/{n}'
	if row < 29+12*n:
		player, index = divmod(row-29-6*n, 6)
		cost_or_value = (index%2 == 0)
		return f'Reserve {index//2} of player {player}/{n} ' + ('cost' if cost_or_value else 'value')
	return f'unknown row {row}'

def _gen_list_of_different_gems(max_num_gems):
	gems = [ np.array([int(i==c) for i in range(7)], dtype=np.int8) for c in range(5) ]
	results = []
	for n in range(1, max_num_gems+1):
		results += [ sum(comb) for comb in itertools.combinations(gems, n) ]
	return results

def _gen_list_of_3gems_2spec():
	gems2 = _gen_list_of_different_gems(2)
	gems2 = gems2 + (np.array(gems2[:5])*2).tolist()
	results = [gems2[i[1]] + gems2[i[2]]  for i in give_ids3]
	return results

list_different_gems_up_to_3 =  _gen_list_of_different_gems(3)
list_different_gems_up_to_2 =  _gen_list_of_different_gems(2)
list_2specs_gems_up_to3     = _gen_list_of_3gems_2spec() #2種類の組み合わせ(0枚含む)
np_different_gems_up_to_2 = np.array(list_different_gems_up_to_2, dtype=np.int8)
np_different_gems_up_to_3 = np.array(list_different_gems_up_to_3, dtype=np.int8)
np_2specs_gems_up_to_3 = np.array(list_2specs_gems_up_to3, dtype=np.int8)

# cards_symmetries = itertools.permutations(range(4))
cards_symmetries   = [(1, 3, 0, 2), (2, 0, 3, 1), (3, 2, 1, 0)]
reserve_symmetries = [
	[], 					# 0 card in reserve
	[], 					# 1 card
	[(1, 0, 2)],			# 2 cards
	[(1, 2, 0), (2, 0, 1)], # 3 cards
]
reserve_symmetries2 = [       # Need constant size to convert to numpy list
	[(-1,-1,-1), (-1,-1,-1)], # 0 card in reserve
	[(-1,-1,-1), (-1,-1,-1)], # 1 card
	[(1, 0, 2) , (-1,-1,-1)], # 2 cards
	[(1, 2, 0) , (2, 0, 1) ], # 3 cards
]
np_cards_symmetries = np.array(cards_symmetries, dtype=np.int8)
np_reserve_symmetries = np.array(reserve_symmetries2, dtype=np.int8)

##### END OF CLASS #####

idx_white, idx_blue, idx_green, idx_red, idx_black, idx_gold, idx_points = range(7)
light_colors = [
	Back.LIGHTWHITE_EX + Fore.BLACK,	# white
	Back.LIGHTBLUE_EX + Fore.WHITE,		# blue
	Back.LIGHTGREEN_EX + Fore.BLACK,	# green
	Back.LIGHTRED_EX + Fore.BLACK,		# red
	Back.LIGHTBLACK_EX + Fore.WHITE,	# black
	Back.LIGHTYELLOW_EX + Fore.BLACK,	# gold
]
strong_colors = [
	Back.WHITE + Fore.BLACK,	# white
	Back.BLUE + Fore.BLACK,		# blue
	Back.GREEN + Fore.BLACK,	# green
	Back.RED + Fore.BLACK,		# red
	Back.BLACK + Fore.WHITE,	# black
	Back.YELLOW + Fore.BLACK,	# gold
]

#    W Blu G  R  Blk  Point
all_nobles = [
	[0, 0, 4, 4, 0, 0, 3],
	[0, 0, 0, 4, 4, 0, 3],
	[0, 4, 4, 0, 0, 0, 3],
	[4, 0, 0, 0, 4, 0, 3],
	[4, 4, 0, 0, 0, 0, 3],
	[3, 0, 0, 3, 3, 0, 3],
	[3, 3, 3, 0, 0, 0, 3],
	[0, 0, 3, 3, 3, 0, 3],
	[0, 3, 3, 3, 0, 0, 3],
	[3, 3, 0, 0, 3, 0, 3],
]
np_all_nobles  = np.array(all_nobles , dtype=np.int8)

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_1 = [
	[
		[[0, 0, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 0, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 2, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 2, 2, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 1, 3, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[1, 0, 1, 2, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 4, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
	],
	[
		[[3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[0, 2, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 0, 1, 0, 2, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 0, 0, 1, 3, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[1, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[2, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
		[[4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
	],
	[
		[[0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[2, 2, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 0, 1, 3, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[1, 2, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]],
		[[0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
	],
	[
		[[0, 3, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 0, 2, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 0, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 2, 2, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[3, 1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 2, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
		[[0, 0, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
	],
	[
		[[0, 0, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[2, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 2, 0, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 1, 0, 2, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 3, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[1, 1, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 0, 0]],
		[[0, 0, 0, 0, 4, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
	],
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_2 = [
	[
		[[0, 2, 2, 3, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 2, 3, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 1]],
		[[0, 5, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[5, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[2, 0, 0, 1, 4, 0, 0], [0, 1, 0, 0, 0, 0, 2]],
		[[0, 6, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 0, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 3, 0, 2, 3, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
		[[0, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[3, 0, 0, 0, 5, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[1, 4, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
		[[0, 0, 0, 6, 0, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
	],
	[
		[[3, 2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[3, 0, 3, 0, 2, 0, 0], [0, 0, 0, 0, 1, 0, 1]],
		[[5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 1, 4, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 2]],
		[[0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
	],
	[
		[[0, 0, 3, 2, 2, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[2, 3, 0, 3, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1]],
		[[0, 0, 0, 5, 0, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 0, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[0, 0, 1, 4, 2, 0, 0], [1, 0, 0, 0, 0, 0, 2]],
		[[6, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
	],
	[
		[[2, 3, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[3, 0, 2, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1]],
		[[0, 0, 5, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 5, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[4, 2, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 2]],
		[[0, 0, 6, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
	]
]

#             COST                      GAIN
#         W Blu G  R  Blk        W Blu G  R  Blk  Point
all_cards_3 = [
	[
		[[3, 0, 3, 3, 5, 0, 0], [0, 1, 0, 0, 0, 0, 3]],
		[[7, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[6, 3, 0, 0, 3, 0, 0], [0, 1, 0, 0, 0, 0, 4]],
		[[7, 3, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 5]],
	],
	[
		[[3, 5, 3, 0, 3, 0, 0], [0, 0, 0, 1, 0, 0, 3]],
		[[0, 0, 7, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 3, 6, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 4]],
		[[0, 0, 7, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0, 5]],
	],
	[
		[[3, 3, 5, 3, 0, 0, 0], [0, 0, 0, 0, 1, 0, 3]],
		[[0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 3, 6, 3, 0, 0], [0, 0, 0, 0, 1, 0, 4]],
		[[0, 0, 0, 7, 3, 0, 0], [0, 0, 0, 0, 1, 0, 5]],
	],
	[
		[[0, 3, 3, 5, 3, 0, 0], [1, 0, 0, 0, 0, 0, 3]],
		[[0, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 3, 6, 0, 0], [1, 0, 0, 0, 0, 0, 4]],
		[[3, 0, 0, 0, 7, 0, 0], [1, 0, 0, 0, 0, 0, 5]],
	],
	[
		[[5, 3, 0, 3, 3, 0, 0], [0, 0, 1, 0, 0, 0, 3]],
		[[0, 7, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[3, 6, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 4]],
		[[0, 7, 3, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 5]],
	]
]

all_cards = [all_cards_1, all_cards_2, all_cards_3]
np_all_cards_1 = np.array(all_cards_1, dtype=np.int8)
np_all_cards_2 = np.array(all_cards_2, dtype=np.int8)
np_all_cards_3 = np.array(all_cards_3, dtype=np.int8)
len_all_cards = np.array([len(all_cards_1[0]), len(all_cards_2[0]), len(all_cards_3[0])], dtype=np.int8)

def _print_round_and_scores(board):
	n = board.num_players
	print('='*10*n, f' round {board.get_round()}    ', end='')
	for p in range(n):
		print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
	print('='*10*n, Style.RESET_ALL)

def _print_nobles(board):
	print(f'{Style.BRIGHT}Nobles:  {Style.RESET_ALL}', end='')
	for noble in board.nobles:
		if noble[idx_points] == 0:
			print(f'< {Style.DIM}empty{Style.RESET_ALL} >', end=' ')
		else:
			print(f'< {noble[idx_points]} points ', end='')
			for i, color in enumerate(light_colors):
				if noble[i] != 0:
					print(f'{color} {noble[i]} {Style.RESET_ALL} ', end='')
			print(f'> ', end='')
	print(f'{Style.RESET_ALL}')

def _print_card_line(card, line, space_between):
	if card[1,:5].sum() == 0:
		print(f' '*(8+space_between), end='')
		return
	card_color = np.flatnonzero(card[1,:5] != 0)[0]
	background = light_colors[card_color]
	print(background, end= '')
	if line == 0:
		print(f'     {Style.BRIGHT}{card[1][idx_points]}{Style.NORMAL}  ', end='')
	else:
		card_cost = np.flatnonzero(card[0,:5] != 0)
		if line-1 < card_cost.size:
			color = card_cost[line-1]
			value = card[0,color]
			print(f' {light_colors[color]} {value} {background}    ', end='')
		else:
			print(' '*8, end='')
	print(Style.RESET_ALL, end=' '*space_between)

def _print_tiers(board):
	for tier in range(2, -1, -1):
		for line in range(5):
			if line == 3:
				print(f'Tier {tier}:  ', end='')
			elif line == 4 :
				print(f'  ({board.nb_deck_tiers[2*tier].sum():>2})   ', end='')
			else:
				print(f'         ', end='')
			for i in range(4):
				_print_card_line(board.cards_tiers[8*tier+2*i:8*tier+2*i+2, :], line, 4)
			print()
		print()

def _print_bank(board):
	print(f'{Style.BRIGHT}Bank: {Style.RESET_ALL}   ', end='')
	for c in range(6):
		print(f'{light_colors[c]} {board.bank[0][c]} {Style.RESET_ALL} ', end='')
	print(f'{Style.RESET_ALL}')

def _print_players(board):
	n = board.num_players
	# NAMES
	print(' '*19, end='')
	for p in range(n):
		print(f'Player {p}', end='')
		if p < n-1:
			print(f' '*26, end='')
	print()

	# NOBLES
	print(' '*9, end='')
	for p in range(n):
		for noble in board.players_nobles[3*p:3*p+3]:
			if noble[idx_points] > 0:
				print(f'  < {Style.BRIGHT}{noble[idx_points]}{Style.RESET_ALL} >  ', end='')
			else:
				print(f'        ', end='')
		print(f' '*10, end='')
	print()

	# GEMS
	print(f'{Style.BRIGHT}Gems: {Style.RESET_ALL}   ', end='')
	for p in range(n):
		for c in range(6):
			my_gems  = board.players_gems[p][c]
			print(f'{light_colors[c]} {my_gems} {Style.RESET_ALL} ', end='')
		print(f' Σ{board.players_gems[p].sum():2}      ', end='')
	print()

	# CARDS
	# print()
	print(f'{Style.BRIGHT}Cards: {Style.RESET_ALL}  ', end='')
	for p in range(n):
		for c in range(5):
			my_cards = board.players_cards[p][c]
			print(f'{light_colors[c]} {my_cards} {Style.RESET_ALL} ', end='')
		print(f'              ', end='')
	print()

	# RESERVED
	if board.players_reserved.sum() > 0:
		print()
		for line in range(5):
			if line == 2:
				print(f'{Style.BRIGHT}Reserve: {Style.RESET_ALL}', end='')
			else:
				print(' '*9, end='')
			for p in range(n):
				for r in range(3):
					reserved = board.players_reserved[6*p+2*r:6*p+2*r+2]
					if reserved[0].sum() != 0:
						_print_card_line(reserved, line, 2)
					else:
						print(f' '*10, end='')
				print(f' '*4, end='')
			print()

def print_board(board):
	_print_round_and_scores(board)
	_print_nobles(board)
	print()
	_print_tiers(board)
	_print_bank(board)
	print()
	_print_players(board)
