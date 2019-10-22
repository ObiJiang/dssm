import torch
import torchvision
import numpy as np

# single DSSM layer
from single_layer import SingleLayerDSSMForMnist
# classification
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# misc
from misc import AttrDict
from tqdm import tqdm
import argparse

# set random seed
random_seed = 1
torch.manual_seed(random_seed)

class DSSM():
	def __init__(self, args):
		config = AttrDict()

		input_dims = [28, 28, 1]
		config.input_dims = np.array(input_dims)
		config.num_layers = 1

		# strides
		config.strides = [2]
		assert(len(config.strides) == config.num_layers)

		# nps
		config.nps = [4]
		assert(len(config.nps) == config.num_layers)

		# dist thresholds
		config.io_ths = [4]
		config.lateral_ths = [0]

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
		config.num_epochs = 2
		config.batch_size = args.batch_size

		# data
		config.data_dir = args.data_dir

		self.config = config

		self.layers = {}

		# mnist train data lodaer
		self.train_loader = torch.utils.data.DataLoader(
		  torchvision.datasets.MNIST(self.config.data_dir, train=True, download=True,
		                             transform=torchvision.transforms.Compose([
		                               torchvision.transforms.ToTensor(),
		                               torchvision.transforms.Normalize(
		                                 (0.1307,), (0.3081,))
		                             ])),
		  batch_size=self.config.batch_size, shuffle=True)

		# mnist test data lodaer
		self.test_loader = torch.utils.data.DataLoader(
		  torchvision.datasets.MNIST(self.config.data_dir, train=False, download=True,
		                             transform=torchvision.transforms.Compose([
		                               torchvision.transforms.ToTensor(),
		                               torchvision.transforms.Normalize(
		                                 (0.1307,), (0.3081,))
		                             ])),
		  batch_size=self.config.batch_size, shuffle=True)

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

	def init_layers_states(self, batch_size):
		for layer_ind in range(self.config.num_layers):
			self.layers[layer_ind].initialize(batch_size)

	def train(self, inp, epoch):
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
					cur_feedback = torch.zeros(int(np.prod(self.config.num_units[-1])), device = self.config.device)
				else: 
					cur_feedback = self.layers[layer_ind + 1].feedback() 
				
				delta[layer_ind] = self.layers[layer_ind].dynamics(cur_inp, cur_feedback, update_step)

			# Hugo's parameter
			if (delta< 1e-4).all() :
				conversion_ticker += 1
				break

		for layer_ind in range(self.config.num_layers):
			if layer_ind == 0:
				cur_inp = inp
			else:
				cur_inp = self.layers[layer_ind - 1].get_output()
			self.layers[layer_ind].plasticity_update(cur_inp, epoch)

		return delta.sum(), conversion_ticker

	def test(self, inp):
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
					cur_feedback = torch.zeros(int(np.prod(self.config.num_units[-1])), device = self.config.device)
				else: 
					cur_feedback = self.layers[layer_ind + 1].feedback() 
				
				delta[layer_ind] = self.layers[layer_ind].dynamics(cur_inp, cur_feedback, update_step)

			# Hugo's parameter
			if (delta< 1e-4).all() :
				conversion_ticker += 1
				break

			hid_vec = self.layers[self.config.num_layers-1].get_output().cpu().data

		return hid_vec

	def run(self):
		for epoch in tqdm(range(self.config.num_epochs)):
			loss = 0
			conversion_ticker = 0
			for idx, (image, label) in enumerate(tqdm(self.train_loader)):
				batch_size = image.shape[0] # input batch size
				image = image.to(self.config.device) # move to device
				image = image.view([batch_size, -1]) # reshape to vectors

				# init states u and v based on input batch size
				self.init_layers_states(batch_size)

				loss_per_image, conversion_ticker_per_image = self.train(image, epoch)
				# TODO: add real loss
				loss += loss_per_image
				conversion_ticker += conversion_ticker_per_image
			
			print("{:} Epoch: loss {:}".format(epoch, loss))

	def classify(self):
		fea_dim = self.layers[self.config.num_layers-1].get_output_dim()
		train_X = np.zeros((60000, fea_dim))
		train_Y = np.zeros(60000)
		test_X = np.zeros((10000, fea_dim))
		test_Y = np.zeros(10000)

		# get training data
		start_save_idx = 0
		for idx, (image, label) in enumerate(tqdm(self.train_loader)):
			batch_size = image.shape[0] # input batch size
			image = image.to(self.config.device) # move to device
			image = image.view([batch_size, -1]) # reshape to vectors

			# init states u and v based on input batch size
			self.init_layers_states(batch_size)
			fea = self.test(image)

			train_X[start_save_idx: start_save_idx + batch_size, :] = fea
			train_Y[start_save_idx: start_save_idx + batch_size] = label
			start_save_idx += batch_size

		print("finishing getting training data ...")

		# get test data
		start_save_idx = 0
		for idx, (image, label) in enumerate(tqdm(self.test_loader)):
			batch_size = image.shape[0] # input batch size
			image = image.to(self.config.device) # move to device
			image = image.view([batch_size, -1]) # reshape to vectors

			# init states u and v based on input batch size
			self.init_layers_states(batch_size)
			fea = self.test(image)

			test_X[start_save_idx: start_save_idx + batch_size, :] = fea
			test_Y[start_save_idx: start_save_idx + batch_size] = label
			start_save_idx += batch_size

		print("finishing getting test data ...")

		clf = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
		clf.fit(train_X, train_Y)

		predicated_labels = clf.predict(test_X)

		acc = accuracy_score(test_Y, predicated_labels)
		print("Accuracy: {}".format(acc))

if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser()

	# training_parameters
	parser.add_argument('--batch_size', default=1, type=int)

	# train/test
	parser.add_argument('--train', default=False, action='store_true')

	# save/load model
	parser.add_argument('--model_save_dir', default='./save.pickle')
	parser.add_argument('--data_dir', default='./data')

	config = parser.parse_args()

if config.train:
	model = DSSM(config)
	model.create_network()
	model.run()
	# save model
	torch.save(vars(model), config.model_save_dir)
	model.classify()

else:
	load_dict = torch.load(config.model_save_dir)
	model = DSSM(config)
	model.__dict__.update(load_dict)
	model.classify()



