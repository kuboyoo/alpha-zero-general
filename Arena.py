import logging
log = logging.getLogger(__name__)

import bisect
from tqdm import tqdm
import numpy as np
from pathlib import Path

from MCTS import MCTS
from splendor.NNet import NNetWrapper as NNet
from utils import *
from time import sleep
import os

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, player3, game, args, display=None, no_record=False):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.player3 = player3
        self.game = game
        self.display = display
        self.lag = args.lag
        self.record_dir = None if no_record == True else Path(args.record_dir)
        
        #以下, 改造箇所
        """
        nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
        self.net = NNet(game, nn_args) #盤面評価用NN

        model_name = "./result_230410/checkpoint_41.pt"
        nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
        net = NNet(game, nn_args)
        cpt_dir, cpt_file = os.path.split(model_name)
        additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
        mcts_args = dotdict({
            'numMCTSSims'     : args.numMCTSSims if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
            'cpuct'           : args.cpuct       if args.cpuct       else additional_keys.get('cpuct'      , 1.0),
            'prob_fullMCTS'   : 1.,
            'forced_playouts' : False,
            'no_mem_optim'    : False,
        })
        self.mcts = MCTS(game, net, mcts_args)
        """
        
        if self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True) #盤面ログ保存用
            self.record_path = self.record_dir.joinpath("record.txt") #全対局テキストファイル

    def playGame(self, verbose=False, other_way=False, cur_player=None, board=None):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # if NUMBER_PLAYERS == 2:
        #     players = [self.player2, self.player1]                             if other_way else [self.player1, self.player2]
        # elif NUMBER_PLAYERS == 3:
        #     players = [self.player2, self.player1, self.player1]               if other_way else [self.player1, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 4:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 5:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        if not other_way:
            #players = [self.player1]+[self.player2]*(self.game.getNumberOfPlayers()-1)
            
            if self.player3 is None:
                players = [self.player1, self.player2]
            else:
                players = [self.player1, self.player2, self.player3]
        else:
            players = [self.player2]+[self.player1]*(self.game.getNumberOfPlayers()-1)
        if cur_player is None:
            curPlayer = 0
            board = self.game.getInitBoard()
        else:
            curPlayer = cur_player
        it = 0
        while not self.game.getGameEnded(board, curPlayer).any():
            it += 1
            if verbose:
                if self.display:
                    self.display(board)
                    if self.lag:
                        sleep(1.0)
                print()
                
                if self.record_dir is not None:
                    savePkl(self.record_dir.joinpath("board_turn_%02d.pkl" % it), board)
                    #game = self.game
                    #savePkl(self.record_dir.joinpath(f"game_turn_{it}.pkl"), game)

                """
                valids = self.game.getValidMoves(board, 0)
                pi, value = self.net.predict(board, valids)
                print(f'Value: {value}')
                """

                """
                #AIの行動確率を表示する場合
                prob = self.mcts.getActionProb(board, temp=1, force_full_search=True, bias=None)[0]
                prob = np.array(prob)
                idx = np.argsort(prob)[::-1][:5] #AI行動確率>0の上位5候補手

                print("AI predicts:")
                for i, p in zip(idx, prob[idx]):
                    print("[%d]: %2.1f%%: %s" % (i, p*100, self.game.moveToString(i, curPlayer)))
                """


                print()
                print(f'Turn {it} Player {curPlayer}: ', end='')
                
            canonical_board = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer](canonical_board)
            valids = self.game.getValidMoves(canonical_board, 0)

            if verbose:
                print(f'P{curPlayer} decided to {self.game.moveToString(action, curPlayer)}')

            if valids[action] == 0:
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            if self.display:
                self.display(board)
            print("Game over: Turn ", str(it), "Result ", self.game.getGameEnded(board, curPlayer))

        #終局盤面もログ出力
        it += 1
        if self.record_dir is not None:
            savePkl(self.record_dir.joinpath("board_turn_%02d.pkl" % it), board)

        MCTS.reset_all_search_trees()
            
        return self.game.getGameEnded(board, curPlayer)[0]

    def playGames(self, num, verbose=False, cur_player=None, board=None):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        ratio_boundaries = [        1-0.60,        1-0.55,        0.55,        0.60         ]
        colors           = ['#d60000',     '#d66b00',     '#f9f900',   '#a0d600',  '#6b8e00'] #https://icolorpalette.com/ff3b3b_ff9d3b_ffce3b_ffff3b_ceff3b

        oneWon, twoWon, draws = 0, 0, 0
        t = tqdm(range(num), desc="Arena.playGames", ncols=120, disable=None)
        for i in t:
            # Since trees may not be resetted, the first games (1vs2) can't be
            # considered as fair as the last games (2vs1). Switching between 
            # 1vs2 and 2vs1 like below seems more fair:
            # 1 2 2 1   1 2 2 1  ...
            one_vs_two = (i%4 == 0) or (i%4 == 3)
            t.set_description('Arena ' + ('(1 vs 2)' if one_vs_two else '(2 vs 1)'), refresh=False)
            gameResult = self.playGame(verbose=verbose, other_way=not one_vs_two, cur_player=cur_player)
            if gameResult == (1. if one_vs_two else -1.):
                oneWon += 1
            elif gameResult == (-1. if one_vs_two else 1.):
                twoWon += 1
            else:
                draws += 1

            t.set_postfix(one_wins=oneWon, two_wins=twoWon, refresh=False)
            ratio = oneWon / (oneWon+twoWon) if oneWon+twoWon>0 else 0.5
            t.colour = colors[bisect.bisect_right(ratio_boundaries, ratio)]
        t.close()

        return oneWon, twoWon, draws
