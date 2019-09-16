import torch

# misc
from misc import AttrDict

class DSSM():
	def __init__(self):
		config = AttrDict()

		config.hidden_units = [10, 10]
		config.num_layers = len(config.hidden_units) - 1
		config.gamma = 0.01

		config.host = torch.device("cpu")
		config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		config.step_size = 0.00001
		config.lr = 0.001

		self.config = config

		self.layers = {}

	def create_network(self):
		for layer_ind in range(self.config.num_layers):
			self.layers[layer_ind] = SingleLayerDSSM(layer_ind, 
													self.config.hidden_units[layer_ind], 
													self.config.hidden_units[layer_ind + 1], 
													self.config)

	def run(self):
		inp = torch.randn(10, device=self.config.device)
		while True:
			# neural dynamics loop
			while True:
				err = 0.0
				for layer_ind in range(self.config.num_layers):
					cur_inp = inp if layer_ind == 0 else self.layers[layer_ind - 1].get_output()
					cur_feedback = torch.zeros(self.config.hidden_units[-1]) if layer_ind == self.config.num_layers - 1 else self.layers[layer_ind + 1].feedback() 
					
					err += self.layers[layer_ind].dynamics(cur_inp, cur_feedback)

				if err < 1e-5:
					break

			for layer_ind in range(self.config.num_layers):
				cur_inp = inp if layer_ind == 0 else self.layers[layer_ind - 1].get_output()
				self.layers[layer_ind].plasticity_update(cur_inp)

model = DSSM()
model.create_network()
model.run()



