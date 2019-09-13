import torch

# misc
from misc import AttrDict

class SingleLayerDSSM():
	def __init__(self, layer_ind, input_dim, hidden_dim, config):
		self.config = config
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.layer_ind = layer_ind

		# weights
		self.W = torch.randn(input_dim, hidden_dim, device = config.device)
		self.L = torch.randn(hidden_dim, hidden_dim, device = config.device)

		# activations
		self.u = torch.zeros(hidden_dim)
		self.r = self.activation(self.u)

		# connectivies
		self.c_W_hat = torch.ones(input_dim, hidden_dim) # there is supposed to be reciprocal
		self.c_P_hat = torch.ones(hidden_dim, hidden_dim)

		# feedback_parameter
		self.feedback_parameter = 0 if self.layer_ind == self.config.num_layers - 1 else 1

	def dynamics(self, prev_layer, feedback):
		r_save = self.r.clone()
		du = - self.u + self.W @ prev_layer - (self.L - torch.eye(self.hidden_dim)) @ self.r + feedback
		self.u += self.config.step_size * du
		self.r = self.activation(self.u)

		err = torch.dist(self.r, r_save, 2)
		return err.item()

	def plasticity_update(self, prev_layer):
		update_step = self.config.lr * self.config.gamma ** (self.layer_ind - self.config.num_layers)

		dW = update_step * (prev_layer.t() @ self.r * torch.sign(self.c_W_hat) - self.W * self.c_W_hat)
		dL = update_step / 2 * (self.r.t() @ self.r * torch.sign(self.c_P_hat) 
			- self.L * self.c_P_hat / (1 + self.config.gamma * self.feedback_parameter))

		self.W += dW
		self.L += dL

	def feedback(self):
		return self.config.gamma * self.W @ self.r

	def activation(self, u):
		r = torch.max(torch.min(u, torch.ones_like(u)), torch.zeros_like(u))
		return r

	def get_output(self):
		return self.r


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



