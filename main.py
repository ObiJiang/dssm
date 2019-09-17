import torch
import torchvision
import numpy as np

# single DSSM layer
from single_layer import SingleLayerDSSMForMnist

# misc
from misc import AttrDict
from tqdm import tqdm
import argparse

# set random seed
random_seed = 1
torch.manual_seed(random_seed)

class DSSM():
	def __init__(self):
		config = AttrDict()

		input_dims = [28, 28, 1]
		config.input_dims = np.array(input_dims)
		config.num_layers = 2

		# strides
		config.strides = [2, 1]
		assert(len(config.strides) == config.num_layers)

		# nps
		config.nps = [4, 16]
		assert(len(config.nps) == config.num_layers)

		# dist thresholds
		config.io_ths = [2, 4]
		config.lateral_ths = [0, 4]

		# set up number of units per layer
		config.num_units = [config.input_dims]
		next_num_units = config.input_dims
		for i in range(config.num_layers):
			next_num_units = (next_num_units/config.strides[i]).astype(int)
			next_num_units[-1] = config.nps[i]
			assert(next_num_units.all() > 0) # must be positive
			config.num_units.append(next_num_units)

		config.gamma = 0.00 # feedback parameter

		# host and device
		config.host = torch.device("cpu")
		config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		# learning rate and step size
		config.euler_lr = 0.2
		config.lr = 1e-2
		config.lr_floor=1e-4
		config.decay = 2

		# training 
		config.num_epochs = 10
		config.batch_size = 300

		self.config = config

		self.layers = {}

	def create_network(self):
		for layer_ind in range(self.config.num_layers):
			mnist_config = AttrDict()
			# layer-wise configs
			mnist_config.layer_ind = layer_ind
			mnist_config.stride = self.config.strides[layer_ind]
			mnist_config.input_dims = self.config.num_units[layer_ind]
			mnist_config.output_dims = self.config.num_units[layer_ind + 1]
			mnist_config.stride = self.config.strides[layer_ind]
			mnist_config.io_th = self.config.io_ths[layer_ind]
			mnist_config.lateral_th = self.config.lateral_ths[layer_ind]

			#network-wise config
			mnist_config.network_config = self.config

			self.layers[layer_ind] = SingleLayerDSSMForMnist(mnist_config)

	def single_pass(self, inp, epoch):
		# neural dynamics loop
		delta = np.ones(self.config.num_layers) * np.inf
		conversion_ticker = 0
		for update_step in range(3000):

			for layer_ind in range(self.config.num_layers):
				# set layer input
				if layer_ind == 0:
					cur_inp = inp
				else:
					cur_inp = self.layers[layer_ind - 1].get_output()

				# set layer feedback input from top layer
				if layer_ind == self.config.num_layers - 1:
					cur_feedback = torch.zeros(int(np.prod(self.config.num_units[-1])))
				else: 
					cur_feedback = self.layers[layer_ind + 1].feedback() 
				
				delta[layer_ind] = self.layers[layer_ind].dynamics(cur_inp, cur_feedback, update_step)

			# Hugo's parameter
			if delta.any() < 1e-4:
				conversion_ticker += 1
				break

		for layer_ind in range(self.config.num_layers):
			if layer_ind == 0:
				cur_inp = inp
			else:
				cur_inp = self.layers[layer_ind - 1].get_output()
			self.layers[layer_ind].plasticity_update(cur_inp, epoch)

		return delta.sum(), conversion_ticker

	def run(self):
		# mnist data lodaer
		train_loader = torch.utils.data.DataLoader(
		  torchvision.datasets.MNIST('./data', train=True, download=True,
		                             transform=torchvision.transforms.Compose([
		                               torchvision.transforms.ToTensor(),
		                               torchvision.transforms.Normalize(
		                                 (0.1307,), (0.3081,))
		                             ])),
		  batch_size=self.config.batch_size, shuffle=True)

		for epoch in tqdm(range(self.config.num_epochs)):
			loss = 0
			conversion_ticker = 0
			for idx, (image, label) in enumerate(train_loader):
				image = image.to(self.config.device)
				image = image.view([self.config.batch_size, -1])

				loss_per_image, conversion_ticker_per_image = self.single_pass(image, epoch)
				loss += loss_per_image
				conversion_ticker += conversion_ticker_per_image
				break
			
			print("{:} Epoch: loss {:}".format(epoch, loss))

if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser()

	# train/test
	parser.add_argument('--train', default=False, action='store_true')

	# save/load model
	parser.add_argument('--model_save_dir', default='./save.pickle')

	config = parser.parse_args()

if config.train:
	model = DSSM()
	model.create_network()
	model.run()

	# save model
	torch.save(vars(model), config.model_save_dir)
else:
	load_dict = torch.load(config.model_save_dir)
	model = DSSM()
	model.__dict__.update(load_dict)
	model.run()



