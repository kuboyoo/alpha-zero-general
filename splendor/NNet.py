import sys
sys.path.append('../../')
from GenericNNetWrapper import GenericNNetWrapper
from .SplendorNNet import SplendorNNet as nn_model

class NNetWrapper(GenericNNetWrapper):
	def init_nnet(self, game, nn_args, use_exchange=False):
		self.nnet = nn_model(game, nn_args, use_token_exchange=use_exchange)
