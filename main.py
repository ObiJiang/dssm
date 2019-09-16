import torch
import numpy as np
# single DSSM layer
from single_layer import SingleLayerDSSMForMnist

# misc
from misc import AttrDict

class DSSM():
	def __init__(self):
		config = AttrDict()

		input_dims = [10, 10, 3]
		config.input_dims = np.array(input_dims)
		config.num_layers = 2

		# strides
		config.strides = [2, 1]
		assert(len(config.strides) == config.num_layers)

		# nps
		config.nps = [4, 16]
		assert(len(config.nps) == config.num_layers)

		# set up number of units per layer
		config.num_units = [config.input_dims]
		next_num_units = config.input_dims
		for i in range(config.num_layers):
			next_num_units = (next_num_units/config.strides[i]).astype(int)
			next_num_units[-1] = config.nps[i]
			assert(next_num_units.all() > 0) # must be positive
			config.num_units.append(next_num_units)

		config.gamma = 0.01 # feedback parameter

		# host and device
		config.host = torch.device("cpu")
		config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		# learning rate and step size
		config.step_size = 0.00001
		config.lr = 0.001

		self.config = config

		self.layers = {}

	def create_network(self):
		for layer_ind in range(self.config.num_layers):
			mnist_config = AttrDict()
			# layer-wise configs
			mnist_config.layer_ind = layer_ind
			mnist_config.input_dims = self.config.num_units[layer_ind]
			mnist_config.output_dims = self.config.num_units[layer_ind + 1]
			mnist_config.stride = self.config.strides[layer_ind]

			#network-wise config
			mnist_config.network_config = self.config

			self.layers[layer_ind] = SingleLayerDSSMForMnist(mnist_config)

	def run(self):
		inp = torch.randn(10, device=self.config.device)
		while True:
			# neural dynamics loop
			while True:
				err = 0.0
				for layer_ind in range(self.config.num_layers):
					# set layer input
					if layer_ind == 0:
						cur_inp = inp
					else:
						cur_inp = self.layers[layer_ind - 1].get_output()

					# set layer feedback input from top layer
					if layer_ind == self.config.num_layers - 1:
						cur_feedback = torch.zeros(self.config.hidden_units[-1])  
					else: 
						cur_feedback = self.layers[layer_ind + 1].feedback() 
					
					err += self.layers[layer_ind].dynamics(cur_inp, cur_feedback)

				if err < 1e-5:
					break

			for layer_ind in range(self.config.num_layers):
				if layer_ind == 0:
					cur_inp = inp
				else:
					cur_inp = self.layers[layer_ind - 1].get_output()
				self.layers[layer_ind].plasticity_update(cur_inp)

model = DSSM()
# model.create_network()
# model.run()



