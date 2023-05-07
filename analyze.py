import numpy as np
from scipy.stats import entropy
import os
import re

from MCTS import MCTS
from splendor.SplendorGame import SplendorGame as Game
from splendor.NNet import NNetWrapper as NNet
from utils import *
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
  
def review(canonical_board, game, curPlayer, net, mcts, temp=0):
  
  valids = game.getValidMoves(canonical_board, 0)
  _, value = net.predict(canonical_board, valids) #形勢評価値

  prob = mcts.getActionProb(canonical_board, temp=temp, force_full_search=True, bias=None)[0]
  prob = np.array(prob)
  idx = np.argsort(prob)[::-1][:5] #AI行動確率>0の上位5候補手

  ent = entropy(prob)
  print(prob[idx])

  #print("AI Value: ", value)

  #print("AI Suggest:")
  #for i, p in zip(idx, prob[idx]):
  #    print("[%d]: %2.1f%%: %s" % (i, p*100, game.moveToString(i, curPlayer)))

  return value[curPlayer], ent

def main(args):
  num_players = 2
  model_name = "./results/result_230428/checkpoint_9.pt" #暫定最強
  record_dir = Path(args.record_dir)
  output_dir = Path(args.output_dir)
  output_path = output_dir.joinpath("report.csv")
  os.makedirs(output_dir, exist_ok=True)

  data = []
  game = Game(num_players)

  nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1)
  net = NNet(game, nn_args)
  cpt_dir, cpt_file = os.path.split(model_name)
  additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
  cpuct = 2.5
  fpu = 0.3
  num_mcts=1200
  mcts_args = dotdict({
    'numMCTSSims'     : num_mcts, # if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
    'cpuct'           : cpuct       if cpuct       else additional_keys.get('cpuct'      , 1.0),
    'fpu'             : fpu,
    'prob_fullMCTS'   : 1.,
    'forced_playouts' : False,
    'no_mem_optim'    : False,
  })
  mcts = MCTS(game, net, mcts_args)

  pkl_paths = list(record_dir.glob("*.pkl"))
  pkl_paths.sort()

  for i, path in enumerate(pkl_paths):
    board = loadPkl(path)
    curPlayer = i % num_players
    
    canonical_board = game.getCanonicalForm(board, curPlayer)
    value, entropy = review(canonical_board, game, curPlayer, net, mcts, temp=1)
    score1 = game.getScore(board, 0)
    score2 = game.getScore(board, 1)

    data.append([i+1, score1, score2, value, entropy])

  df = pd.DataFrame(data, columns=["round", "score1", "score2", "value", "entropy"])
  df.to_csv(output_path, index=False)
  df["value"].plot()
  plt.ylim(-1, 1)
  plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='tester')
  parser.add_argument('--record-dir', '-r' , action='store', default="./record/230416_01/", type=str, help='record directory')
  parser.add_argument('--output-dir', '-o' , action='store', default="./report/230416_01/", type=str, help='report directory')
  args = parser.parse_args()
  main(args)