import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
        self.c_W_hat = self.create_c_W().to(self.network_config.device) # this is supposed to be reciprocal
        self.c_L_hat = self.create_c_L().to(self.network_config.device)

        # Hugo's parameter -> factor
        factor = np.sqrt(((np.sum(self.c_W_hat.cpu().data.numpy())/self.output_dims[-1])/np.prod(self.output_dims[:-1])))

        # weights
        self.W = torch.randn(self.output_linear_dim, self.input_linear_dim, 
            device = self.network_config.device) / factor * (self.c_W_hat)
        self.L = torch.eye(self.output_linear_dim, 
             device = self.network_config.device) * (self.c_L_hat)
        # self.L = torch.randn(self.output_linear_dim, self.output_linear_dim,
        # 	device = self.network_config.device)

        # activations
        self.u = None 
        self.r = None
        self.act_fn = torch.nn.Tanh()

        # feedback_parameter
        if self.layer_ind == self.network_config.num_layers - 1:
            self.feedback_parameter = 0  
        else: 
            self.feedback_parameter = 1

    def state_dict(self):
        return vars(self)

    def initialize(self, batch_size):
        # activations
        self.u = torch.zeros(batch_size, self.output_linear_dim, device = self.network_config.device)
        self.r = self.activation(self.u)

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
        du = - 1 * self.u + prev_layer @ self.W.t() - self.r @ (self.L - torch.eye(self.output_linear_dim, device = self.network_config.device)).t() + feedback

        # Hugo's euler update rule
        lr = max((self.network_config.euler_lr/(1+0.005*step)), 0.05)

        self.u += lr * du
        self.r = self.activation(self.u)
        
        err_all = torch.norm(self.r - r_save, p=2, dim=1)/(1e-10 + torch.norm(r_save, p=2, dim=1))
        err = torch.mean(err_all)
        return err.item()

    def plasticity_update(self, prev_layer,epoch):
        lr = max(self.network_config.lr/(1+self.network_config.decay*epoch), self.network_config.lr_floor)

        if self.network_config.gamma > 0:
            update_step = lr * self.network_config.gamma ** (1 + self.layer_ind - self.network_config.num_layers)
        else:
            update_step = lr

        dW = update_step * (self.r.t() @ prev_layer * (self.c_W_hat) - self.W * self.c_W_hat)
        dL = update_step / 2 * (self.r.t() @ self.r * (self.c_L_hat) 
            - self.L * self.c_L_hat / (1 + self.network_config.gamma * self.feedback_parameter))

        self.W += dW
        self.L += dL

        self.W = self.W * (self.c_W_hat)
        self.L = self.L * (self.c_L_hat)
    
    def plot_weights(self, name):
        plt.figure()
        im = plt.imshow(self.W.cpu().numpy(), aspect='auto', interpolation='none', origin='lower')
        plt.colorbar(im)
        plt.savefig(self.network_config.weight_save_dir + "/"+ name + "_w.png")
        np.save(self.network_config.weight_save_dir + "/"+ name + "_" + "w.npy", self.W.cpu().numpy())

        plt.figure()
        im = plt.imshow(self.L.cpu().numpy(), aspect='auto', interpolation='none', origin='lower')
        plt.colorbar(im)
        plt.savefig(self.network_config.weight_save_dir + "/"+name + "_l.png")
        np.save(self.network_config.weight_save_dir + "/"+ name + "_" + "l.npy", self.L.cpu().numpy())

    def feedback(self):
        return self.network_config.gamma * self.r @ self.W

    def activation(self, u):
        r = torch.max(torch.min(u, torch.ones_like(u, device = self.network_config.device)), 
                 torch.zeros_like(u, device = self.network_config.device))
        # r = self.act_fn(u)
        return r

    def get_output(self):
        return self.r

    def get_output_dim(self):
        return self.output_linear_dim

class SingleLayerDSSMForMnistSpike(SingleLayerDSSMForMnist):
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
        self.c_W_hat = self.create_c_W().to(self.network_config.device) # this is supposed to be reciprocal
        self.c_L_hat = self.create_c_L().to(self.network_config.device)

        # Hugo's parameter -> factor
        factor = np.sqrt(((np.sum(self.c_W_hat.cpu().data.numpy())/self.output_dims[-1])/np.prod(self.output_dims[:-1])))

        # weights
        self.W = torch.randn(self.output_linear_dim, self.input_linear_dim, 
            device = self.network_config.device) / factor * (self.c_W_hat)
        self.L = torch.eye(self.output_linear_dim, 
            device = self.network_config.device) * (self.c_L_hat)

        # feedback_parameter
        if self.layer_ind == self.network_config.num_layers - 1:
            self.feedback_parameter = 0  
        else: 
            self.feedback_parameter = 1

        # spiking params

        # firing rate bounds
        config.lower_bound = 0
        config.upper_bound = 1

        # opt param
        config.lambda_1 = 0.0
        config.lambda_2 = 0.0

        # analog param
        config.dt = 1.0

        # spiking param
        config.dt_s = 0.1
        if config.dt_s >= 1/config.upper_bound:
            config.dt_r = 0
        else:
            config.dt_r = (1/config.upper_bound)/config.dt_s
        
        config.v_r = 0

        self.dt_s = config.dt_s
        self.dt_r = config.dt_r
        self.v_r = config.v_r
        self.lambda_1 = config.lambda_1
        self.lambda_2 = config.lambda_2
        self.nb_units = self.output_linear_dim
        self.v_f_adjustment = 0.0
        
        self.L_diag = torch.diagonal(self.L)
        self.L_hat = self.L - torch.diag(torch.diagonal(self.L)) + self.v_f_adjustment*torch.eye(self.nb_units, device = self.network_config.device)
        
        # The firing rate for neuron 2 should be lambda_2 + L_ii
        self.v_f = config.lambda_2 + self.L_diag - self.v_f_adjustment
        
        if (self.v_f < 0).any():
            raise ValueError('Choose a larger lambda_2')
        
        # not meant to be used before calling init_v_i()
        self.i_s = None
        self.v_s = None
        self.delta = None
        self.spike_count = None
        self.time_tick = None
        self.y = None
    
    def init_v_i(self, x):
        self.i_s = x @ self.W.t() - self.lambda_1
        self.v_s = torch.ones(x.shape[0],self.nb_units, device = self.network_config.device) * self.v_r
        self.delta = torch.zeros(x.shape[0],self.nb_units, device = self.network_config.device)
        self.spike_count = torch.zeros(x.shape[0],self.nb_units, device = self.network_config.device)
        self.y = torch.zeros(x.shape[0],self.nb_units, device = self.network_config.device)
        self.time_tick = 0
        self.refactory_clock = torch.ones(x.shape[0], self.nb_units, device = self.network_config.device) * self.dt_r
    
    def _refactory_counting(self):
        self.refactory_clock -= 1
        self.refactory_clock *= (self.refactory_clock > 0).float()
        
    def _active_mask(self):
        return self.refactory_clock <= 0
    
    def _integ_mask(self):
        integrate_mask = self.v_s <= self.v_f
        
        return integrate_mask
    
    def _fire(self):
        fire_mask = (self.v_s >= self.v_f) * self._active_mask()
        self.delta = fire_mask * 1
        self.spike_count += self.delta.float()
        self.v_s[fire_mask] = self.v_r
        self.refactory_clock[fire_mask] = self.dt_r

    def dynamics(self, x):
        for _ in range(20000):
            self.y_save = self.y.clone()
            self.time_tick += 1

            mask = self._integ_mask()
            self._refactory_counting()
            
            # spike dynamics
            di_s = - self.i_s +  x @ self.W.t() - self.lambda_1 -  (self.delta.float() /self.dt_s) @ self.L_hat.t()
            self.i_s += di_s * self.dt_s * mask.float()
            self.v_s += self.i_s * self.dt_s * mask.float()
            
            self._fire()
            
            self.y = self.spike_count/(self.time_tick*self.dt_s)
            if self.time_tick % 5000 == 0:
                pass
                # print(self.time_tick, self.y)
                # print(self.time_tick, self.y[self.y > 0])
                # err_all = torch.norm(self.y - self.y_save, p=2, dim=1)/(1e-10 + torch.norm( self.y_save, p=2, dim=1))
                # err = torch.mean(err_all)
                # print(self.time_tick, err.item())

            if self.time_tick > 10000:
                pass
                # err_all = torch.norm(self.y - self.y_save, p=2, dim=1)/(1e-10 + torch.norm( self.y_save, p=2, dim=1))
                # err = torch.mean(err_all)

                # if err < 5e-4:
                #     print(self.time_tick, err.item())
                #     print(self.time_tick, self.y[self.y > 0])
                #     break

        return self.y

    
    def plasticity_update(self, prev_layer, epoch):
        lr = max(self.network_config.lr/(1+self.network_config.decay*epoch), self.network_config.lr_floor)

        if self.network_config.gamma > 0:
            update_step = lr * self.network_config.gamma ** (1 + self.layer_ind - self.network_config.num_layers)
        else:
            update_step = lr

        dW = update_step * (self.y.t() @ prev_layer * (self.c_W_hat) - self.W * self.c_W_hat)
        dL = update_step / 2 * (self.y.t() @ self.y * (self.c_L_hat) 
            - self.L * self.c_L_hat / (1 + self.network_config.gamma * self.feedback_parameter))

        self.W += dW
        self.L += dL

        self.W = self.W * (self.c_W_hat)
        self.L = self.L * (self.c_L_hat)

        self.L_diag = torch.diagonal(self.L)
        self.L_hat = self.L - torch.diag(torch.diagonal(self.L)) + self.v_f_adjustment*torch.eye(self.nb_units, device = self.network_config.device)
        
        # The firing rate for neuron 2 should be lambda_2 + L_ii
        self.v_f = self.lambda_2 + self.L_diag - self.v_f_adjustment
        
    def get_output(self):
        return self.y

    def get_output_dim(self):
        return self.output_linear_dim
