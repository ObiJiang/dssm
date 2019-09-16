import torch

class SingleLayerDSSMForMnist():
	def __init__(self, config):
		self.input_dims = config.input_dims
		self.ouput_dims = config.hidden_dims
		self.layer_ind = config.layer_ind

		self.network_config = config.network_config

		# weights
		self.W = torch.randn(input_dim, hidden_dim, device = self.network_config.device)
		self.L = torch.randn(hidden_dim, hidden_dim, device = self.network_config.device)

		# activations
		self.u = torch.zeros(hidden_dim)
		self.r = self.activation(self.u)

		# connectivies
		self.c_W_hat = torch.ones(input_dim, hidden_dim) # this is supposed to be reciprocal
		self.c_L_hat = torch.ones(hidden_dim, hidden_dim)

		# feedback_parameter
		self.feedback_parameter = 0 if self.layer_ind == self.config.num_layers - 1 else 1

	def create_c_W(self):
		pass

	def create_c_L(self):
		pass

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
		dL = update_step / 2 * (self.r.t() @ self.r * torch.sign(self.c_L_hat) 
			- self.L * self.c_L_hat / (1 + self.config.gamma * self.feedback_parameter))

		self.W += dW
		self.L += dL

	def feedback(self):
		return self.config.gamma * self.W @ self.r

	def activation(self, u):
		r = torch.max(torch.min(u, torch.ones_like(u)), torch.zeros_like(u))
		return r

	def get_output(self):
		return self.r
