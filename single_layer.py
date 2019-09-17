import torch
import numpy as np
import itertools

class SingleLayerDSSMForMnist():
	"""
	Note: All the weights are reshaped into 1-D vector for calculation,
	but the connectivity is specified by c_W_hat and c_L_hat (1/c for non-zero elements)
	"""
	def __init__(self, config):
		self.input_dims = config.input_dims
		self.input_linear_dim = int(np.prod(self.input_dims))
		self.output_dims = config.output_dims
		self.output_linear_dim = int(np.prod(self.output_dims))

		self.layer_ind = config.layer_ind
		self.stride = config.stride
		self.io_th = config.io_th
		self.lateral_th = config.lateral_th

		self.network_config = config.network_config

		# connectivies
		self.c_W_hat = self.create_c_W() # this is supposed to be reciprocal
		self.c_L_hat = self.create_c_L()

		# Hugo's parameter -> factor
		factor = np.sqrt(((np.sum(self.c_W_hat.data.numpy())/self.output_dims[-1])/np.prod(self.output_dims[:-1])))

		# weights
		self.W = torch.randn(self.output_linear_dim, self.input_linear_dim, 
			device = self.network_config.device) / factor
		self.L = torch.eye(self.output_linear_dim,
		 	device = self.network_config.device)
		# self.L = torch.randn(self.output_linear_dim, self.output_linear_dim,
		# 	device = self.network_config.device)

		# activations
		self.u = torch.zeros(self.network_config.batch_size, self.output_linear_dim)
		self.r = self.activation(self.u)

		# feedback_parameter
		if self.layer_ind == self.network_config.num_layers - 1:
			self.feedback_parameter = 0  
		else: 
			self.feedback_parameter = 1

	def create_c_W(self):
		"""
		compute once, so it may not be highly optimized.
		the mask is the same for all channel pairs
		"""
		in_nps = self.input_dims[-1]
		out_nps = self.output_dims[-1]
		stride_padding = self.stride/2
		fea_dim_ind = len(self.input_dims)-1

		single_mask_dims = np.concatenate([self.output_dims[:-1], self.input_dims[:-1]])
		c_W = torch.zeros(list(single_mask_dims))

		# create indices matrix for easy dist calculation
		input_indices = []
		for dim_ind, dim in np.ndenumerate(self.input_dims[:-1]):
			repeat_vec = (np.ones(len(self.input_dims)) * self.input_dims).astype(int)
			repeat_vec[dim_ind] = 1
			repeat_vec[-1] = 1

			view_vec = np.ones(len(self.input_dims)).astype(int)
			view_vec[dim_ind] = dim

			indices_per_dim = torch.arange(dim).view(list(view_vec)).repeat(list(repeat_vec))
			input_indices.append(indices_per_dim)

		# Hugo Parameter -> 0.5
		input_indices_tensor = torch.cat(input_indices, dim = fea_dim_ind) + 0.5
		
		# calculate C for one channel pair
		for idx in itertools.product(*[range(s) for s in self.output_dims[:-1]]):
			center = torch.tensor(idx) * self.stride + stride_padding
			dist_mat = torch.norm(input_indices_tensor-center.float(), p=2, dim = fea_dim_ind)
			c_W[idx] = (dist_mat <= self.io_th)

		# replicate C for all channel pairs
		nps_repeat_vec = np.ones(len(self.input_dims) + len(self.output_dims)).astype(int)
		nps_repeat_vec[len(self.output_dims)-1] = out_nps
		nps_repeat_vec[len(self.input_dims)-1 + len(self.output_dims)] = in_nps

		nps_view_vec = np.concatenate([self.output_dims, self.input_dims]).astype(int)
		nps_view_vec[len(self.output_dims)-1] = 1
		nps_view_vec[len(self.input_dims)-1 + len(self.output_dims)] = 1
		
		c_W_tile = c_W.view(list(nps_view_vec)).repeat(list(nps_repeat_vec))
		return c_W_tile.view([self.output_linear_dim, self.input_linear_dim])


	def create_c_L(self):
		"""
		compute once, so it may not be highly optimized
		"""
		out_nps = self.output_dims[-1]
		stride_padding = self.stride/2
		fea_dim_ind = len(self.output_dims)-1

		single_mask_dims = np.concatenate([self.output_dims[:-1], self.output_dims[:-1]])
		c_L = torch.zeros(list(single_mask_dims))

		# create indices matrix for easy dist calculation
		output_indices = []
		for dim_ind, dim in np.ndenumerate(self.output_dims[:-1]):
			repeat_vec = (np.ones(len(self.output_dims)) * self.output_dims).astype(int)
			repeat_vec[dim_ind] = 1
			repeat_vec[-1] = 1

			view_vec = np.ones(len(self.output_dims)).astype(int)
			view_vec[dim_ind] = dim

			indices_per_dim = torch.arange(dim).view(list(view_vec)).repeat(list(repeat_vec))
			output_indices.append(indices_per_dim)

		output_indices_tensor = torch.cat(output_indices, dim = fea_dim_ind)

		for idx in itertools.product(*[range(s) for s in self.output_dims[:-1]]):
			center = torch.tensor(idx)
			dist_mat = torch.norm(output_indices_tensor-center.float(), p=2, dim = fea_dim_ind)
			c_L[idx] = (dist_mat <= self.lateral_th)

		# replicate C for all channel pairs
		nps_repeat_vec = np.ones(len(self.output_dims) + len(self.output_dims)).astype(int)
		nps_repeat_vec[len(self.output_dims)-1] = out_nps
		nps_repeat_vec[len(self.output_dims)-1 + len(self.output_dims)] = out_nps

		nps_view_vec = np.concatenate([self.output_dims, self.output_dims]).astype(int)
		nps_view_vec[len(self.output_dims)-1] = 1
		nps_view_vec[len(self.output_dims)-1 + len(self.output_dims)] = 1
		
		c_L_tile = c_L.view(list(nps_view_vec)).repeat(list(nps_repeat_vec))
		return c_L_tile.view([self.output_linear_dim, self.output_linear_dim])

	def dynamics(self, prev_layer, feedback, step):
		r_save = self.r.clone()
		du = - self.u + prev_layer @ self.W.t() - self.r @ (self.L - torch.eye(self.output_linear_dim)).t() + feedback
		
		# Hugo's euler update rule
		lr = max((self.network_config.euler_lr/(1+0.005*step)), 0.05)

		self.u += lr * du
		self.r = self.activation(self.u)

		err = torch.dist(self.r, r_save, 2)
		return err.item()

	def plasticity_update(self, prev_layer,epoch):
		lr = max(self.network_config.lr/(1+self.network_config.decay*epoch), self.network_config.lr_floor)

		update_step = lr * self.network_config.gamma ** (self.layer_ind - self.network_config.num_layers)

		dW = update_step * (self.r.t() @ prev_layer * torch.sign(self.c_W_hat) - self.W * self.c_W_hat)
		dL = update_step / 2 * (self.r.t() @ self.r * torch.sign(self.c_L_hat) 
			- self.L * self.c_L_hat / (1 + self.network_config.gamma * self.feedback_parameter))

		self.W += dW
		self.L += dL

	def feedback(self):
		return self.network_config.gamma * self.r @ self.W.t() 

	def activation(self, u):
		r = torch.max(torch.min(u, torch.ones_like(u)), torch.zeros_like(u))
		return r

	def get_output(self):
		return self.r
