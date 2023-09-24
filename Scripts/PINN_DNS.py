import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Data_Handler import *
from Directory import *
from Visualisation import *

### PINN MODEL ###

class Vanilla_Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, nb_hidden_layers: int, min_max_input: torch.Tensor, mean_output: torch.Tensor, std_output: torch.Tensor):
        super().__init__()
        
        # parameters and layers
        self.nb_hidden_layers = nb_hidden_layers

        self.enc = nn.Linear(input_dim, d_model, bias=True)
        self.layers = nn.ModuleList(nn.Linear(d_model, d_model, bias=True) for i in range(nb_hidden_layers-1))
        self.dec = nn.Linear(d_model, output_dim, bias=False)
        
        # input/output scaling factors
        self.min_max_input = min_max_input
        self.mean_output = mean_output
        self.std_output = std_output

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rescale inputs
        x[:,0] = 2*(x[:,0]-self.min_max_input[0]) / (self.min_max_input[1]-self.min_max_input[0]) - 1
        x[:,1] = 2*(x[:,1]-self.min_max_input[2]) / (self.min_max_input[3]-self.min_max_input[2]) - 1
        x[:,2] = 2*(x[:,2]-self.min_max_input[4]) / (self.min_max_input[5]-self.min_max_input[4]) - 1
        x[:,3] = 2*(x[:,3]-self.min_max_input[6]) / (self.min_max_input[7]-self.min_max_input[6]) - 1
        
        # forward
        y = self.activation(self.enc(x))
        for i in range(self.nb_hidden_layers-1):
            y = self.activation(self.layers[i](y))
        y = self.dec(y)
        
        # rescale outputs
        y[:,0] = self.mean_output[0] + self.std_output[0]*y[:,0]
        y[:,1] = self.mean_output[1] + self.std_output[1]*y[:,1]
        y[:,2] = self.mean_output[2] + self.std_output[2]*y[:,2]
        y[:,3] = self.mean_output[3] + self.std_output[3]*y[:,3]
        y[:,4] = self.mean_output[4] + self.std_output[4]*y[:,4]
        return y
    


class PINN_Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, nb_hidden_layers: int, criterion: nn.Module, alpha: float, lr: float, 
                 folder_dir: str, folder_name: str, device: any, min_max_input: torch.Tensor, mean_output: torch.Tensor, std_output: torch.Tensor, noise_level: int or float, min_rel_err_noise: torch.Tensor):
        super().__init__()
        
        # instantiate device
        self.device = device

        # instantiate input/output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # instantiate model width and depth
        self.d_model = d_model
        self.nb_hidden_layers = nb_hidden_layers
        
        # instantiate input/output scaling factors
        self.min_max_input = min_max_input.to(self.device)
        self.mean_output = mean_output.to(self.device)
        self.std_output = std_output.to(self.device)
            
        # instantiate model
        self.model = Vanilla_Model(input_dim=self.input_dim, output_dim=self.output_dim, d_model=self.d_model, nb_hidden_layers=self.nb_hidden_layers, min_max_input=self.min_max_input, mean_output=self.mean_output, std_output=self.std_output)
        self.model = self.model.to(self.device)
        
        # instantiate loss criterion & L_data prefactor alpha 
        self.criterion = criterion
        self.MSELoss = nn.MSELoss(reduction = 'sum')
        self.L1Loss = nn.L1Loss(reduction = 'sum')

        # instantiate lr values
        self.lr = lr

        # instantiate directories for results
        self.folder_dir = folder_dir
        self.folder_name = folder_name

        # instantiate epoch
        self.epoch = torch.tensor(0).to(self.device)
        self.list_epoch = torch.tensor([]).to(self.device)

        # instantiate tensors ("lists") of losses (for loss plots)

        # per batch

        # l2 loss for optimization
        self.L_l_uz_batch = torch.tensor([]).to(self.device)    # bc
        self.L_l_ur_batch = torch.tensor([]).to(self.device)    # bc
        self.L_l_uth_batch = torch.tensor([]).to(self.device)   # bc
        self.L_l_p_batch = torch.tensor([]).to(self.device)     # bc
        self.L_l_temp_batch = torch.tensor([]).to(self.device)  # entire domain

        self.L_l_e1_batch = torch.tensor([]).to(self.device)    # heat equation
        self.L_l_e2_batch = torch.tensor([]).to(self.device)    # momentum z
        self.L_l_e3_batch = torch.tensor([]).to(self.device)    # momentum r
        self.L_l_e4_batch = torch.tensor([]).to(self.device)    # momentum th
        self.L_l_e5_batch = torch.tensor([]).to(self.device)    # continuity

        # per epoch
    
        # l2 loss for optimization
        self.L_l_uz = torch.tensor([]).to(self.device)    # bc
        self.L_l_ur = torch.tensor([]).to(self.device)    # bc
        self.L_l_uth = torch.tensor([]).to(self.device)   # bc
        self.L_l_p = torch.tensor([]).to(self.device)     # bc
        self.L_l_temp = torch.tensor([]).to(self.device)  # entire domain

        self.L_l_e1 = torch.tensor([]).to(self.device)    # heat equation
        self.L_l_e2 = torch.tensor([]).to(self.device)    # momentum z
        self.L_l_e3 = torch.tensor([]).to(self.device)    # momentum r
        self.L_l_e4 = torch.tensor([]).to(self.device)    # momentum th
        self.L_l_e5 = torch.tensor([]).to(self.device)    # continuity

        self.L_l_data = torch.tensor([]).to(self.device)
        self.L_l_eq = torch.tensor([]).to(self.device)
        self.L_l_tot = torch.tensor([]).to(self.device)

        # loss prefactor alpha
        self.alpha = alpha

        # instantiate absolute train errors (for error plots)
        self.abs_err_v_train = torch.tensor([]).to(self.device) # [epoch_eval,3]
        self.abs_err_p_train = torch.tensor([]).to(self.device) # [epoch_eval]
        self.abs_err_temp_train = torch.tensor([]).to(self.device) # [epoch_eval]

        # instantiate relative train errors (for error plots)
        self.rel_err_v_train = torch.tensor([]).to(self.device) # [epoch_eval,3]
        self.rel_err_p_train = torch.tensor([]).to(self.device) # [epoch_eval]
        self.rel_err_temp_train = torch.tensor([]).to(self.device) # [epoch_eval]

        # instantiate list_epoch_eval (for error plots) 
        self.list_epoch_eval = torch.tensor([]).to(self.device) # [epoch_eval]

        # create directories
        self.folder_model, self.folder_loss, self.folder_error, self.folder_scalar, self.folder_vector = create_directories(dir_path=self.folder_dir, folder_name=self.folder_name)
        

        # initialize time_step k and zone for progressive algo
        self.k = torch.tensor(0).to(self.device)
        self.zone = torch.tensor(0).to(self.device)

        # store min_rel_err_noise
        self.noise_level = noise_level
        self.min_rel_err_noise = min_rel_err_noise.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x=x)
    

    def init_param(self, init_method: str = 'Xavier' or 'Normal'):

        if init_method not in ('Xavier', 'Normal'):
            print('default initialization')

        else: 
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        if init_method == 'Xavier':
                            nn.init.xavier_uniform_(param, gain= nn.init.calculate_gain(nonlinearity='tanh'))
                        else:
                            nn.init.normal_(param)
    
    

    def init_optimizer(self, optimizer: str = 'Adam' or 'SGD'):
        with torch.no_grad():
            self.group_enc = nn.ParameterList([])
            self.group_layers = nn.ParameterList([])
            self.group_dec = nn.ParameterList([])

            for name, param in self.model.named_parameters():
                if name.startswith('enc'):
                    self.group_enc.append(param)
                elif name.startswith('layers'):
                    self.group_layers.append(param)
                else:
                    self.group_dec.append(param)

            if optimizer not in ('Adam', 'SGD') or optimizer == 'Adam':
                print('default optimizer : Adam')
                self.optimizer = torch.optim.Adam([
                    {'params': self.group_layers, 'name': 'group_layers'},
                    {'params': self.group_enc, 'lr': self.lr, 'name': 'group_enc'},
                    {'params': self.group_dec, 'lr': self.lr, 'name': 'group_dec'}], lr=self.lr, betas=(0.9, 0.99), eps=1e-8)
            else:
                self.optimizer = torch.optim.SGD([
                    {'params': self.group_layers, 'name': 'group_layers'},
                    {'params': self.group_enc, 'lr': self.lr, 'name': 'group_enc'},
                    {'params': self.group_dec, 'lr': self.lr, 'name': 'group_dec'}], lr=self.lr)
    

    def init_scheduler(self, scheduler: str = 'ExponentialLR' or 'LinearLR' or 'MultistepLR' or None):
        with torch.no_grad():

            if scheduler not in ('ExponentialLR', 'LinearLR', 'MultistepLR', None) or scheduler == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
            elif scheduler == None:
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)
            elif scheduler == 'LinearLR':
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0.01, total_iters=1000)
            else: 
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000, 5000, 10000, 50000], gamma=0.3)


    def abs_err_data(self, out_model: torch.Tensor, v_tgt_data: torch.Tensor, p_tgt_data: torch.Tensor, temp_tgt_data: torch.Tensor, d_0: int or float, a_0: int or float, rho_0: int or float, temp_0: int or float) -> tuple:
        l_uz = (out_model[:,0].unsqueeze(-1) - v_tgt_data[:,0].unsqueeze(-1)) * a_0     # [m/s]
        l_uz = self.L1Loss(l_uz, torch.zeros_like(l_uz).to(self.device))

        l_ur = (out_model[:,1].unsqueeze(-1) - v_tgt_data[:,1].unsqueeze(-1)) * a_0     # [m/s]
        l_ur = self.L1Loss(l_ur, torch.zeros_like(l_ur).to(self.device))

        l_uth = (out_model[:,2].unsqueeze(-1) - v_tgt_data[:,2].unsqueeze(-1)) * a_0    # [m/s]
        l_uth = self.L1Loss(l_uth, torch.zeros_like(l_uth).to(self.device))

        l_p = (out_model[:,3].unsqueeze(-1) - p_tgt_data) * rho_0 * a_0 * a_0 / 1000    # [kPa]
        l_p = self.L1Loss(l_p, torch.zeros_like(l_p).to(self.device))

        l_temp = (out_model[:,4].unsqueeze(-1) - temp_tgt_data) * temp_0                # [K]
        l_temp = self.L1Loss(l_temp, torch.zeros_like(l_temp).to(self.device))

        return l_uz, l_ur, l_uth, l_p, l_temp


    def rel_err_data(self, out_model: torch.Tensor, v_tgt_data: torch.Tensor, p_tgt_data: torch.Tensor, temp_tgt_data: torch.Tensor) -> tuple:
        eps = 1e-8
        
        l_uz = (out_model[:,0].unsqueeze(-1) - v_tgt_data[:,0].unsqueeze(-1)) / (torch.std(v_tgt_data[:,0].unsqueeze(-1)) + eps)
        l_uz = 100*self.MSELoss(l_uz, torch.zeros_like(l_uz).to(self.device))

        l_ur = (out_model[:,1].unsqueeze(-1) - v_tgt_data[:,1].unsqueeze(-1)) / (torch.std(v_tgt_data[:,1].unsqueeze(-1)) + eps)
        l_ur = 100*self.MSELoss(l_ur, torch.zeros_like(l_ur).to(self.device))

        l_uth = (out_model[:,2].unsqueeze(-1) - v_tgt_data[:,2].unsqueeze(-1)) / (torch.std(v_tgt_data[:,2].unsqueeze(-1)) + eps)
        l_uth = 100*self.MSELoss(l_uth, torch.zeros_like(l_uth).to(self.device))

        l_p = (out_model[:,3].unsqueeze(-1) - p_tgt_data) / (torch.std(p_tgt_data) + eps)
        l_p = 100*self.MSELoss(l_p, torch.zeros_like(l_p).to(self.device))

        l_temp = (out_model[:,4].unsqueeze(-1) - temp_tgt_data) / (torch.std(temp_tgt_data) + eps)
        l_temp = 100*self.MSELoss(l_temp, torch.zeros_like(l_temp).to(self.device))

        return l_uz, l_ur, l_uth, l_p, l_temp


    def loss_data(self, out_model: torch.Tensor, temp_tgt_data: torch.Tensor) -> tuple:
        eps = 1e-8
        l_temp = (out_model[:,4].unsqueeze(-1) - temp_tgt_data) / (self.std_output[4] + eps)
        l_temp = self.criterion(l_temp, torch.zeros_like(l_temp).to(self.device))

        return l_temp


    def loss_bc(self, out_model: torch.Tensor, v_tgt_bc: torch.Tensor, p_tgt_bc: torch.Tensor) -> tuple:
        eps = 1e-8
        
        l_uz = (out_model[:,0].unsqueeze(-1) - v_tgt_bc[:,0].unsqueeze(-1)) / (self.std_output[0] + eps)
        l_uz = self.criterion(l_uz, torch.zeros_like(l_uz).to(self.device))

        l_ur = (out_model[:,1].unsqueeze(-1) - v_tgt_bc[:,1].unsqueeze(-1)) / (self.std_output[1] + eps)
        l_ur = self.criterion(l_ur, torch.zeros_like(l_ur).to(self.device))

        l_uth = (out_model[:,2].unsqueeze(-1) - v_tgt_bc[:,2].unsqueeze(-1)) / (self.std_output[2] + eps)
        l_uth = self.criterion(l_uth, torch.zeros_like(l_uth).to(self.device))
                
        l_p = (out_model[:,3].unsqueeze(-1) - p_tgt_bc) / (self.std_output[3] + eps)
        l_p = self.criterion(l_p, torch.zeros_like(l_p).to(self.device))

        return l_uz, l_ur, l_uth, l_p
    

    def loss_eq(self, input_eq: torch.Tensor, r_eq: torch.Tensor, th_eq: torch.Tensor, z_eq: torch.Tensor, out_model: torch.Tensor, Re: torch.Tensor, Pr: torch.Tensor) -> tuple:
        ## Outputs and their derivatives
        uz = out_model[:,0].unsqueeze(-1)
        ur = out_model[:,1].unsqueeze(-1)
        uth = out_model[:,2].unsqueeze(-1)
        p_eq = out_model[:,3].unsqueeze(-1)
        T_eq = out_model[:,4].unsqueeze(-1)
 
        # derivatives of uz
        uz_z_r_th_t = torch.autograd.grad(outputs=uz, inputs=input_eq, grad_outputs=torch.ones_like(uz), retain_graph=True, create_graph=True)[0] # [N, 4]
        uz_z = uz_z_r_th_t[:,0].unsqueeze(-1) # [N, 1]
        uz_r = uz_z_r_th_t[:,1].unsqueeze(-1) # [N, 1]
        uz_th = uz_z_r_th_t[:,2].unsqueeze(-1) # [N, 1]
        uz_t = uz_z_r_th_t[:,3].unsqueeze(-1) # [N, 1]
        uz_zz = torch.autograd.grad(outputs=uz_z, inputs=z_eq, grad_outputs=torch.ones_like(uz_z), retain_graph=True, create_graph=True)[0] # [N, 1]
        uz_rr = torch.autograd.grad(outputs=uz_r, inputs=r_eq, grad_outputs=torch.ones_like(uz_r), retain_graph=True, create_graph=True)[0] # [N, 1]
        uz_thth = torch.autograd.grad(outputs=uz_th, inputs=th_eq, grad_outputs=torch.ones_like(uz_th), retain_graph=True, create_graph=True)[0] # [N, 1]
                
        # derivatives of ur
        ur_z_r_th_t = torch.autograd.grad(outputs=ur, inputs=input_eq, grad_outputs=torch.ones_like(ur), retain_graph=True, create_graph=True)[0] # [N, 4]
        ur_z = ur_z_r_th_t[:,0].unsqueeze(-1) # [N, 1]
        ur_r = ur_z_r_th_t[:,1].unsqueeze(-1) # [N, 1]
        ur_th = ur_z_r_th_t[:,2].unsqueeze(-1) # [N, 1]
        ur_t = ur_z_r_th_t[:,3].unsqueeze(-1) # [N, 1]
        ur_zz = torch.autograd.grad(outputs=ur_z, inputs=z_eq, grad_outputs=torch.ones_like(ur_z), retain_graph=True, create_graph=True)[0] # [N, 1]
        ur_rr = torch.autograd.grad(outputs=ur_r, inputs=r_eq, grad_outputs=torch.ones_like(ur_r), retain_graph=True, create_graph=True)[0] # [N, 1]
        ur_thth = torch.autograd.grad(outputs=ur_th, inputs=th_eq, grad_outputs=torch.ones_like(ur_th), retain_graph=True, create_graph=True)[0] # [N, 1]
        
        # derivatives of uth
        uth_z_r_th_t = torch.autograd.grad(outputs=uth, inputs=input_eq, grad_outputs=torch.ones_like(uth), retain_graph=True, create_graph=True)[0] # [N, 4]
        uth_z = uth_z_r_th_t[:,0].unsqueeze(-1) # [N, 1]
        uth_r = uth_z_r_th_t[:,1].unsqueeze(-1) # [N, 1]
        uth_th = uth_z_r_th_t[:,2].unsqueeze(-1) # [N, 1]
        uth_t = uth_z_r_th_t[:,3].unsqueeze(-1) # [N, 1]
        uth_zz = torch.autograd.grad(outputs=uth_z, inputs=z_eq, grad_outputs=torch.ones_like(uth_z), retain_graph=True, create_graph=True)[0] # [N, 1]
        uth_rr = torch.autograd.grad(outputs=uth_r, inputs=r_eq, grad_outputs=torch.ones_like(uth_r), retain_graph=True, create_graph=True)[0] # [N, 1]
        uth_thth = torch.autograd.grad(outputs=uth_th, inputs=th_eq, grad_outputs=torch.ones_like(uth_th), retain_graph=True, create_graph=True)[0] # [N, 1]

        # derivatives of p
        p_z = torch.autograd.grad(outputs=p_eq, inputs=z_eq, grad_outputs=torch.ones_like(p_eq), retain_graph=True, create_graph=True)[0]
        p_r = torch.autograd.grad(outputs=p_eq, inputs=r_eq, grad_outputs=torch.ones_like(p_eq), retain_graph=True, create_graph=True)[0]
        p_th = torch.autograd.grad(outputs=p_eq, inputs=th_eq, grad_outputs=torch.ones_like(p_eq), retain_graph=True, create_graph=True)[0]

        # derivatives of T
        T_z_r_th_t = torch.autograd.grad(outputs=T_eq, inputs=input_eq, grad_outputs=torch.ones_like(T_eq), retain_graph=True, create_graph=True)[0] # [N, 4]
        T_z = T_z_r_th_t[:,0].unsqueeze(-1) # [N, 1]
        T_r = T_z_r_th_t[:,1].unsqueeze(-1) # [N, 1]
        T_th = T_z_r_th_t[:,2].unsqueeze(-1) # [N, 1]
        T_t = T_z_r_th_t[:,3].unsqueeze(-1) # [N, 1]
        T_zz = torch.autograd.grad(outputs=T_z, inputs=z_eq, grad_outputs=torch.ones_like(T_z), retain_graph=True, create_graph=True)[0] # [N, 1]
        T_rr = torch.autograd.grad(outputs=T_r, inputs=r_eq, grad_outputs=torch.ones_like(T_r), retain_graph=True, create_graph=True)[0] # [N, 1]
        T_thth = torch.autograd.grad(outputs=T_th, inputs=th_eq, grad_outputs=torch.ones_like(T_th), retain_graph=True, create_graph=True)[0] # [N, 1]

        # factor
        eps = 1e-8
        self.factor_e1 = torch.mean(torch.abs(T_t)) + eps
        self.factor_e2 = torch.mean(torch.abs(uz_t)) + eps
        self.factor_e3 = torch.mean(torch.abs(ur_t)) + eps
        self.factor_e4 = torch.mean(torch.abs(uth_t)) + eps
        self.factor_e5 = torch.mean(torch.abs(torch.sqrt(uz_z**2+ur_r**2+(1/r_eq*uth_th)**2))) + eps

        ## Residuals
        Pe = Pr*Re

        # heat equation
        e1 = (T_t + ur*T_r + 1/r_eq*uth*T_th + uz*T_z) - 1/Pe*(1/r_eq*T_r + T_rr + 1/(r_eq)**2*T_thth + T_zz) # [N, 1]
        e1 = e1/self.factor_e1
        e1 = self.criterion(e1, torch.zeros_like(e1).to(self.device))

        # momentum equations
        e2 = (uz_t + ur*uz_r + 1/r_eq*uth*uz_th + uz*uz_z + p_z) - 1/Re*(1/r_eq*uz_r + uz_rr + 1/(r_eq)**2*uz_thth + uz_zz) # [N, 1]
        e2 = e2/self.factor_e2
        e2 = self.criterion(e2, torch.zeros_like(e2).to(self.device))

        e3 = (ur_t + ur*ur_r + 1/r_eq*uth*ur_th + uz*ur_z - uth**2/r_eq + p_r) - 1/Re*(1/r_eq*ur_r + ur_rr + 1/(r_eq)**2*(ur_thth - ur - 2*uth_th) + ur_zz) # [N, 1]
        e3 = e3/self.factor_e3
        e3 = self.criterion(e3, torch.zeros_like(e3).to(self.device))

        e4 = (uth_t + ur*uth_r + 1/r_eq*uth*uth_th + uz*uth_z + ur*uth/r_eq + 1/r_eq*p_th) - 1/Re*(1/r_eq*uth_r + uth_rr + 1/(r_eq)**2*(uth_thth - uth + 2*ur_th) + uth_zz) # [N, 1]
        e4 = e4/self.factor_e4
        e4 = self.criterion(e4, torch.zeros_like(e4).to(self.device))

        # continuity equation
        e5 = ur_r + ur/r_eq + 1/r_eq*uth_th + uz_z # [N, 1]
        e5 = e5/self.factor_e5
        e5 = self.criterion(e5, torch.zeros_like(e5).to(self.device))

        return e1, e2, e3, e4, e5


    # plot residuals
    def plot_l_ei(self, dir_path: str):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.plot(self.list_epoch.tolist(), self.L_l_e1.tolist(), 'k-', label='Heat_eq')
        ax.plot(self.list_epoch.tolist(), self.L_l_e2.tolist(), 'r-', label='Momentum_z')
        ax.plot(self.list_epoch.tolist(), self.L_l_e3.tolist(), 'm-', label='Momentum_r')
        ax.plot(self.list_epoch.tolist(), self.L_l_e4.tolist(), 'y-', label='Momentum_th')
        ax.plot(self.list_epoch.tolist(), self.L_l_e5.tolist(), 'c-', label='Continuity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Residual Loss ' + f"(\u03B1 = {self.alpha:.2f})")
        ax.set_ylabel(r'$L_{res}$', fontsize=14)
        ax.set_xlabel(r'epoch')
        ax.legend(loc='lower left')
        ax.grid(visible=True, which='major', color='grey', linestyle='-')
        ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
        plt.savefig(dir_path + 'l_ei.png')
        print('...saved l_ei.png')
        plt.close()
  

    # plot data losses
    def plot_l_data(self, dir_path: str, algo: str):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.plot(self.list_epoch.tolist(), self.L_l_temp.tolist(), 'k-', label='temperature')
        if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
            ax.plot(self.list_epoch.tolist(), self.L_l_uz.tolist(), 'r-', label='uz_bc')
            ax.plot(self.list_epoch.tolist(), self.L_l_ur.tolist(), 'm-', label='ur_bc')
            ax.plot(self.list_epoch.tolist(), self.L_l_uth.tolist(), 'y-', label='uth_bc')
            ax.plot(self.list_epoch.tolist(), self.L_l_p.tolist(), 'c-', label='pressure_bc')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Data Loss ' + f"(\u03B1 = {self.alpha:.2f})")
        ax.set_ylabel(r'$L_{data}$', fontsize=14)
        ax.set_xlabel(r'epoch')
        ax.legend(loc='lower left')
        ax.grid(visible=True, which='major', color='grey', linestyle='-')
        ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
        plt.savefig(dir_path + 'l_data.png')
        print('...saved l_data.png')
        plt.close()


    # plot overall data, eq, total losses
    def plot_loss(self, dir_path: str):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.plot(self.list_epoch.tolist(), self.L_l_data.tolist(), 'g-', label='data loss')
        ax.plot(self.list_epoch.tolist(), self.L_l_eq.tolist(), 'b-', label='residual loss')
        ax.plot(self.list_epoch.tolist(), self.L_l_tot.tolist(), 'k-', label='total loss: ' + f"\u03B1= {self.alpha:.2f}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Total Loss ' + f"(\u03B1 = {self.alpha:.2f})")
        ax.set_ylabel(r'$L_{tot}$', fontsize=14)
        ax.set_xlabel(r'epoch')
        ax.legend(loc='lower left')
        ax.grid(visible=True, which='major', color='grey', linestyle='-')
        ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
        plt.savefig(dir_path + 'loss.png')
        print('...saved loss.png')
        plt.close()


    # plot train & test errors (rel. or abs.) for uz, ur, uth, p, T
    def plot_error_i(self, ax, i, rel_err: bool):
        if rel_err == True:
            ax[i].plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,i].tolist(), 'b-', label='train')
            ax[i].set_xlabel(r'epoch')
            ax[i].set_ylabel(r'$L_{2}$  Rel. Error  [%]')
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].legend(loc='lower left')
            ax[i].grid(visible=True, which='major', color='grey', linestyle='-')
            ax[i].grid(visible=True, which='minor', color='lightgrey', linestyle='--')
        else:
            ax[i].plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,i].tolist(), 'b-', label='train')
            ax[i].set_xlabel(r'epoch')
            ax[i].set_ylabel(r'$L_{1}$  Abs. Error')
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].legend(loc='lower left')
            ax[i].grid(visible=True, which='major', color='grey', linestyle='-')
            ax[i].grid(visible=True, which='minor', color='lightgrey', linestyle='--')


    def plot_error(self, dir_path: str, rel_err: bool):
        if rel_err == True:
            fig, ax = plt.subplots(5, 1, figsize=(10,20), sharex='all')
            fig.tight_layout(pad=7.0, h_pad=3.0)
            self.rel_err_train = torch.cat((self.rel_err_v_train, self.rel_err_p_train, self.rel_err_temp_train), dim=-1)   #[epoch_eval, 5]
            for i in range(5):
                self.plot_error_i(ax=ax, i=i, rel_err=rel_err)
            ax[0].set_title('Relative Error Uz ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[1].set_title('Relative Error Ur ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[2].set_title('Relative Error Uth ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[3].set_title('Relative Error Pressure ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[4].set_title('Relative Error Temperature ' + f"(\u03B1 = {self.alpha:.2f})")
            plt.savefig(dir_path + 'rel_error.png')
            print('...saved rel_error.png')
            plt.close()
        else:
            fig, ax = plt.subplots(5, 1, figsize=(10,20), sharex='all')
            fig.tight_layout(pad=7.0, h_pad=3.0)
            self.abs_err_train = torch.cat((self.abs_err_v_train, self.abs_err_p_train, self.abs_err_temp_train), dim=-1)   #[epoch_eval, 5]
            for i in range(5):
                self.plot_error_i(ax=ax, i=i, rel_err=rel_err)
            ax[0].set_title('Absolute Error Uz ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[1].set_title('Absolute Error Ur ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[2].set_title('Absolute Error Uth ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[3].set_title('Absolute Error Pressure ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[4].set_title('Absolute Error Temperature ' + f"(\u03B1 = {self.alpha:.2f})")
            ax[0].set_ylabel(r'$L_{1}$  Abs. Error [m/s]')
            ax[1].set_ylabel(r'$L_{1}$  Abs. Error [m/s]')
            ax[2].set_ylabel(r'$L_{1}$  Abs. Error [m/s]')
            ax[3].set_ylabel(r'$L_{1}$  Abs. Error [kPa]')
            ax[4].set_ylabel(r'$L_{1}$  Abs. Error [K]')
            plt.savefig(dir_path + 'abs_error.png')
            print('...saved abs_error.png')
            plt.close()
    

    def plot_err(self, dir_path: str, rel_err: bool):
        if rel_err == True:
            self.rel_err_train = torch.cat((self.rel_err_v_train, self.rel_err_p_train, self.rel_err_temp_train), dim=-1)   #[epoch_eval, 5]
            #plot rel_err on same plot
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
            ax.plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,-1].tolist(), 'k-', label='temperature')
            ax.plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,0].tolist(), 'r-', label='uz')
            ax.plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,1].tolist(), 'm-', label='ur')
            ax.plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,2].tolist(), 'y-', label='uth')
            ax.plot(self.list_epoch_eval.tolist(), self.rel_err_train[:,3].tolist(), 'c-', label='pressure')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Relative Error ' + f"(\u03B1 = {self.alpha:.2f})")
            ax.set_ylabel(r'$L_{2}$ Rel. Error [%]')
            ax.set_xlabel(r'epoch')
            ax.legend(loc='lower left')
            ax.grid(visible=True, which='major', color='grey', linestyle='-')
            ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
            plt.savefig(dir_path + 'rel_err.png')
            print('...saved rel_err.png')
            plt.close()

        else:
            self.abs_err_train = torch.cat((self.abs_err_v_train, self.abs_err_p_train, self.abs_err_temp_train), dim=-1)   #[epoch_eval, 5]
            #plot abs_err on same plot
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
            ax.plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,-1].tolist(), 'k-', label='temperature [K]')
            ax.plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,0].tolist(), 'r-', label='uz [m/s]')
            ax.plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,1].tolist(), 'm-', label='ur [m/s]')
            ax.plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,2].tolist(), 'y-', label='uth [m/s]')
            ax.plot(self.list_epoch_eval.tolist(), self.abs_err_train[:,3].tolist(), 'c-', label='pressure [kPa]')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Absolute Error ' + f"(\u03B1 = {self.alpha:.2f})")
            ax.set_xlabel(r'epoch')
            ax.legend(loc='lower left')
            ax.grid(visible=True, which='major', color='grey', linestyle='-')
            ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
            plt.savefig(dir_path + 'abs_err.png')
            print('...saved abs_err.png')
            plt.close()


    # save checkpoint
    def save_checkpoint(self, save_path):
        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),

                    'list_epoch': self.list_epoch,

                    'L_l_uz': self.L_l_uz,
                    'L_l_ur': self.L_l_ur,
                    'L_l_uth': self.L_l_uth,
                    'L_l_p': self.L_l_p,
                    'L_l_temp': self.L_l_temp,

                    'L_l_e1': self.L_l_e1,
                    'L_l_e2': self.L_l_e2,
                    'L_l_e3': self.L_l_e3,
                    'L_l_e4': self.L_l_e4,
                    'L_l_e5': self.L_l_e5,

                    'L_l_data': self.L_l_data,
                    'L_l_eq': self.L_l_eq,
                    'L_l_tot': self.L_l_tot,

                    'list_epoch_eval': self.list_epoch_eval,

                    'abs_err_v_train': self.abs_err_v_train,
                    'abs_err_p_train': self.abs_err_p_train,
                    'abs_err_temp_train': self.abs_err_temp_train,

                    'rel_err_v_train': self.rel_err_v_train,
                    'rel_err_p_train': self.rel_err_p_train,
                    'rel_err_temp_train': self.rel_err_temp_train,

                    'min_rel_err_noise': self.min_rel_err_noise,

                    'k': self.k,
                    'zone': self.zone

                    }, save_path)
        

    # save model parameters
    def save_checkpoint_model(self, save_path):
        torch.save({
                    'epoch': self.list_epoch[-1],
                    'model_state_dict': self.model.state_dict(),
                    }, save_path)
        

    # load model, optimizer and scheduler parameters to proceed simulation
    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Iterate through the checkpoint dictionary and set attributes dynamically
        for key, value in checkpoint.items():
            if key != 'model_state_dict' or key != 'optimizer_state_dict' or key != 'scheduler_state_dict':
                setattr(self, key, value)


    # store training errors (rel. or abs.) for uz, ur, uth, p, temp in tensors ("lists")
    def eval_train(self, b_size_eval: int, input_tensor: torch.Tensor, v_data: torch.Tensor, p_data: torch.Tensor, temp_data: torch.Tensor, idx_train: torch.Tensor, rel_err: bool, d_0: int or float, a_0: int or float, rho_0: int or float, temp_0: int or float,):
        idx = torch.split(idx_train, b_size_eval, dim=0) # [N]
        err_v = torch.tensor([0,0,0]).to(self.device)
        err_p = torch.tensor([0]).to(self.device)
        err_temp = torch.tensor([0]).to(self.device)
        for b in range(len(idx)):
            output_eval = self.model.forward(input_tensor[idx[b]].to(self.device)) # [b_size, 5]
            v_tgt_data = v_data[idx[b],:].to(self.device)
            p_tgt_data = p_data[idx[b],:].to(self.device)
            temp_tgt_data = temp_data[idx[b],:].to(self.device)
            # relative error wrt statistics
            if rel_err == True:
                err_uz_b, err_ur_b, err_uth_b, err_p_b, err_temp_b = self.rel_err_data(out_model=output_eval, v_tgt_data=v_tgt_data, p_tgt_data=p_tgt_data, temp_tgt_data=temp_tgt_data)
            # absolute error
            else:
                err_uz_b, err_ur_b, err_uth_b, err_p_b, err_temp_b = self.abs_err_data(out_model=output_eval, v_tgt_data=v_tgt_data, p_tgt_data=p_tgt_data, temp_tgt_data=temp_tgt_data, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0)

            err_v_b = torch.cat((err_uz_b.unsqueeze(-1), err_ur_b.unsqueeze(-1), err_uth_b.unsqueeze(-1)), dim=0)

            err_v = err_v + err_v_b
            err_p = err_p + err_p_b
            err_temp = err_temp + err_temp_b

        if rel_err == True:
            self.rel_err_v_train = torch.cat((self.rel_err_v_train, 1/idx_train.shape[0]*err_v.unsqueeze(0)), dim=0)
            self.rel_err_p_train = torch.cat((self.rel_err_p_train, 1/idx_train.shape[0]*err_p.unsqueeze(-1)))
            self.rel_err_temp_train = torch.cat((self.rel_err_temp_train, 1/idx_train.shape[0]*err_temp.unsqueeze(-1)))
        else:
            self.abs_err_v_train = torch.cat((self.abs_err_v_train, 1/idx_train.shape[0]*err_v.unsqueeze(0)), dim=0)
            self.abs_err_p_train = torch.cat((self.abs_err_p_train, 1/idx_train.shape[0]*err_p.unsqueeze(-1)))
            self.abs_err_temp_train = torch.cat((self.abs_err_temp_train, 1/idx_train.shape[0]*err_temp.unsqueeze(-1)))


    # training
    def train(self, num_epoch: int, zone_every: int, eval_every: int, save_every: int, b_size: int, b_size_eval: int, input_tensor: torch.Tensor,
              z_data: torch.Tensor, r_data: torch.Tensor, th_data: torch.Tensor, t_data: torch.Tensor, 
              v_data: torch.Tensor, p_data: torch.Tensor, temp_data: torch.Tensor, 
              v_data_noisy: torch.Tensor, p_data_noisy: torch.Tensor, temp_data_noisy: torch.Tensor,
              idx_train: torch.Tensor,
              Re_const: int or float, Pr: int or float, d_0: int or float, a_0: int or float, rho_0: int or float, temp_0: int or float, algo: str):
        

        for epoch in range(1,num_epoch+1):

            # l2 loss for optimization
            self.L_l_uz_batch = torch.tensor([]).to(self.device)    # bc
            self.L_l_ur_batch = torch.tensor([]).to(self.device)    # bc
            self.L_l_uth_batch = torch.tensor([]).to(self.device)   # bc
            self.L_l_p_batch = torch.tensor([]).to(self.device)     # bc
            self.L_l_temp_batch = torch.tensor([]).to(self.device)  # entire domain

            self.L_l_e1_batch = torch.tensor([]).to(self.device)    # heat equation
            self.L_l_e2_batch = torch.tensor([]).to(self.device)    # momentum z
            self.L_l_e3_batch = torch.tensor([]).to(self.device)    # momentum r
            self.L_l_e4_batch = torch.tensor([]).to(self.device)    # momentum th
            self.L_l_e5_batch = torch.tensor([]).to(self.device)    # continuity
            
            with torch.no_grad():
                if algo == 'progressive_spatial_temporal':

                    # initialize
                    if epoch == 1:
                        time_zones = True
                        idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc')
                        idx_data = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, self.zone, self.k, time_zones, condition='data')
                        
                    # add zone every
                    if (self.epoch.item()) % zone_every == 0 and self.epoch.item() !=0 :
                        
                        # add timestep every
                        if (self.epoch.item()) % (zone_every + 3 * zone_every) == 0:
                            if self.k.item() < len(t_data) - 1:
                                self.zone = torch.tensor(0).to(self.device)
                                self.k = self.k + 1
                                idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc')
                                idx_data = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='data')

                        else:
                            if self.k.item() < len(t_data) - 1:
                                self.zone = self.zone + 1
                            else:
                                if self.zone.item() < 3:
                                    self.zone = self.zone + 1

                        idx_data = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, self.zone, self.k, time_zones, condition='data')
                    
                    print('k', self.k.item())
                    print('zone', self.zone.item())
                    print('idx_data', idx_data.shape[0])
                    print('idx_bc', idx_bc.shape[0])


                elif algo == 'progressive_spatial':
                    if epoch == 1:
                        time_zones = False
                        idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc')
                        idx_data = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, self.zone, self.k, time_zones, condition='data')
                    
                    if (self.epoch.item()) % (zone_every) == 0 and (self.epoch.item()) != 0:
                        if self.zone.item() < 3:
                            self.zone = self.zone + 1
                            idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc')
                            idx_data = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, self.zone, self.k, time_zones, condition='data')
                            
                    print('zone', self.zone.item())
                    print('idx_data', idx_data.shape[0])
                    print('idx_bc', idx_bc.shape[0])

                elif algo == 'bc':
                    if epoch == 1:
                        time_zones = False
                        idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc')
                        idx_data = idx_train
                    #print('idx_data', idx_data.shape[0])
                    #print('idx_bc', idx_bc.shape[0])


                elif algo == 'bc_ic':
                    if epoch == 1:
                        time_zones = False
                        idx_bc = get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, 0, self.k, time_zones, condition='bc_ic')
                        idx_data = idx_train

                    print('idx_data', idx_data.shape[0])
                    print('idx_bc_ic', idx_bc.shape[0])

                elif algo == 'temp_only':
                    if epoch == 1:
                        idx_data = idx_train

                    # print('idx_data', idx_data.shape[0])

                # shuffle data set (shuffle inputs and targets) and select batch samples
                idx = torch.randperm(idx_data.shape[0])
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    idx_b = torch.randperm(idx_bc.shape[0])

                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    # split indices to generate mini-batches for both data and bc based on mini-batches of total size b_size
                    # adjust the number of chunks if they differ for idx_data and idx_bc
                    b_size_data = int(b_size * len(idx_data.reshape(-1)) / (len(idx_bc.reshape(-1)) + len(idx_data.reshape(-1))))
                    b_size_bc = int(b_size * len(idx_bc.reshape(-1)) / (len(idx_bc.reshape(-1)) + len(idx_data.reshape(-1))))

                    # if number of chuncks for idx_data and idx_bc are equal then split indices accordingly
                    if len(idx_bc.reshape(-1)) // b_size_bc == len(idx_data.reshape(-1)) // b_size_data:
                        idx = torch.tensor_split(idx, len(idx_data.reshape(-1)) // b_size_data + 1, dim=0) 
                        idx_b = torch.tensor_split(idx_b, len(idx_bc.reshape(-1)) // b_size_bc + 1, dim=0)
                    
                    # if number of chunkcs for idx_data and idx_bc differ, split indices with the maximum nb of chunks
                    else:
                        print('condition')
                        len_max = max((len(idx_bc.reshape(-1)) // b_size_bc, len(idx_data.reshape(-1)) // b_size_data))
                        idx = torch.tensor_split(idx, len_max + 1, dim=0)
                        idx_b = torch.tensor_split(idx_b, len_max + 1, dim=0)

                    # print(k)
                    # print('len_idx', len(idx), 'shape', idx[0].shape)
                    # print('len_idx_bc', len(idx_b), 'shape', idx_b[0].shape)
                
                elif algo == 'temp_only':
                    idx = torch.tensor_split(idx, len(idx_data.reshape(-1)) // b_size + 1, dim=0)
            
            # iterate over each mini-batch
            for b in range(len(idx)):

                self.optimizer.zero_grad()

                with torch.no_grad():
                    z_batch = input_tensor[idx_data[idx[b]],0].unsqueeze(-1).to(self.device)       # [b_size, 1]
                    r_batch = input_tensor[idx_data[idx[b]],1].unsqueeze(-1).to(self.device)       # [b_size, 1]
                    th_batch = input_tensor[idx_data[idx[b]],2].unsqueeze(-1).to(self.device)      # [b_size, 1]
                    t_batch = input_tensor[idx_data[idx[b]],3].unsqueeze(-1).to(self.device)       # [b_size, 1]

                    temp_tgt_data_batch = temp_data_noisy[idx_data[idx[b]],:].to(self.device)            # [b_size, 1]

                    if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                        z_bc_batch = input_tensor[idx_bc[idx_b[b]],0].unsqueeze(-1).to(self.device)     # [b_size, 1]
                        r_bc_batch = input_tensor[idx_bc[idx_b[b]],1].unsqueeze(-1).to(self.device)     # [b_size, 1]
                        th_bc_batch = input_tensor[idx_bc[idx_b[b]],2].unsqueeze(-1).to(self.device)    # [b_size, 1]
                        t_bc_batch = input_tensor[idx_bc[idx_b[b]],3].unsqueeze(-1).to(self.device)     # [b_size, 1]

                        v_tgt_bc_batch = v_data_noisy[idx_bc[idx_b[b]],:].to(self.device)                     # [b_size, 3]
                        p_tgt_bc_batch = p_data_noisy[idx_bc[idx_b[b]],:].to(self.device)                     # [b_size, 1]

                # leaf nodes
                z_batch.requires_grad = True
                r_batch.requires_grad = True
                th_batch.requires_grad = True
                t_batch.requires_grad = True
                
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    z_bc_batch.requires_grad = True
                    r_bc_batch.requires_grad = True
                    th_bc_batch.requires_grad = True
                    t_bc_batch.requires_grad = True
                
                # inputs
                input_batch = torch.cat((z_batch, r_batch, th_batch, t_batch), dim=-1)                  # [b_size, 4]
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    input_bc_batch = torch.cat((z_bc_batch, r_bc_batch, th_bc_batch, t_bc_batch), dim=-1)   # [b_size, 4]

                # forward pass
                out_model_batch = self.forward(input_batch)             # [b_size, 5]
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    out_model_bc_batch = self.forward(input_bc_batch)       # [b_size, 5]

                # loss computation
                l_temp_batch = self.loss_data(out_model=out_model_batch, temp_tgt_data=temp_tgt_data_batch)
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    l_uz_batch, l_ur_batch, l_uth_batch, l_p_batch = self.loss_bc(out_model=out_model_bc_batch, v_tgt_bc=v_tgt_bc_batch, p_tgt_bc=p_tgt_bc_batch)                
                l_e1_batch, l_e2_batch, l_e3_batch, l_e4_batch, l_e5_batch = self.loss_eq(input_eq=input_batch, r_eq=r_batch, th_eq=th_batch, z_eq=z_batch, out_model=out_model_batch, Re=Re_const, Pr=Pr)

                # data loss, residuals, total loss
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    l_data_batch = torch.stack((l_uz_batch, l_ur_batch, l_uth_batch, l_p_batch, l_temp_batch))
                    l_data_batch = torch.sum(l_data_batch)
                elif algo == 'temp_only':
                    l_data_batch = l_temp_batch
                l_eq_batch = torch.stack((l_e1_batch, l_e2_batch, l_e3_batch, l_e4_batch, l_e5_batch))
                l_eq_batch = torch.sum(l_eq_batch)
                l_tot_batch = self.alpha*l_data_batch + (1-self.alpha)*l_eq_batch

                # backpropagation
                l_tot_batch.backward() 

                # storing batch losses in tensor (L_: "list")
                # removing the effect of 'mean' operation from loss computations: multiply each l_batch by corresponding mini-batch size
                # this is done because we might have mini-batches of different sizes (last mini-batch may be smaller when performing split operation)
                with torch.no_grad():

                    # l2 loss used for optimization
                    if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                        self.L_l_uz_batch = torch.cat((self.L_l_uz_batch, z_bc_batch.shape[0]*l_uz_batch.unsqueeze(-1)), dim=0)         #bc
                        self.L_l_ur_batch = torch.cat((self.L_l_ur_batch, z_bc_batch.shape[0]*l_ur_batch.unsqueeze(-1)), dim=0)         #bc
                        self.L_l_uth_batch = torch.cat((self.L_l_uth_batch, z_bc_batch.shape[0]*l_uth_batch.unsqueeze(-1)), dim=0)      #bc
                        self.L_l_p_batch = torch.cat((self.L_l_p_batch, z_bc_batch.shape[0]*l_p_batch.unsqueeze(-1)), dim=0)            #bc
                    self.L_l_temp_batch = torch.cat((self.L_l_temp_batch, z_batch.shape[0]*l_temp_batch.unsqueeze(-1)), dim=0)      #entire domain

                    self.L_l_e1_batch = torch.cat((self.L_l_e1_batch, z_batch.shape[0]*l_e1_batch.unsqueeze(-1)), dim=0)            #entire domain
                    self.L_l_e2_batch = torch.cat((self.L_l_e2_batch, z_batch.shape[0]*l_e2_batch.unsqueeze(-1)), dim=0)            #entire domain
                    self.L_l_e3_batch = torch.cat((self.L_l_e3_batch, z_batch.shape[0]*l_e3_batch.unsqueeze(-1)), dim=0)            #entire domain
                    self.L_l_e4_batch = torch.cat((self.L_l_e4_batch, z_batch.shape[0]*l_e4_batch.unsqueeze(-1)), dim=0)            #entire domain
                    self.L_l_e5_batch = torch.cat((self.L_l_e5_batch, z_batch.shape[0]*l_e5_batch.unsqueeze(-1)), dim=0)            #entire domain

                # update weights
                self.optimizer.step()
                
            # update learning rate
            self.scheduler.step()

            # update epoch
            self.epoch = self.epoch + 1
    
            # store epochs & corresponding losses in tensors
            with torch.no_grad():
                self.list_epoch = torch.cat((self.list_epoch, self.epoch.unsqueeze(-1)), dim=0)

                # l2 losses used for optimization
                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    self.L_l_uz = torch.cat((self.L_l_uz, 1/idx_bc.shape[0]*torch.sum(self.L_l_uz_batch).unsqueeze(-1)), dim=0)             #bc
                    self.L_l_ur = torch.cat((self.L_l_ur, 1/idx_bc.shape[0]*torch.sum(self.L_l_ur_batch).unsqueeze(-1)), dim=0)             #bc
                    self.L_l_uth = torch.cat((self.L_l_uth, 1/idx_bc.shape[0]*torch.sum(self.L_l_uth_batch).unsqueeze(-1)), dim=0)          #bc
                    self.L_l_p = torch.cat((self.L_l_p, 1/idx_bc.shape[0]*torch.sum(self.L_l_p_batch).unsqueeze(-1)), dim=0)                #bc
                self.L_l_temp = torch.cat((self.L_l_temp, 1/idx_data.shape[0]*torch.sum(self.L_l_temp_batch).unsqueeze(-1)), dim=0)    #entire domain

                self.L_l_e1 = torch.cat((self.L_l_e1, 1/idx_data.shape[0]*torch.sum(self.L_l_e1_batch).unsqueeze(-1)), dim=0)          #entire domain
                self.L_l_e2 = torch.cat((self.L_l_e2, 1/idx_data.shape[0]*torch.sum(self.L_l_e2_batch).unsqueeze(-1)), dim=0)          #entire domain
                self.L_l_e3 = torch.cat((self.L_l_e3, 1/idx_data.shape[0]*torch.sum(self.L_l_e3_batch).unsqueeze(-1)), dim=0)          #entire domain
                self.L_l_e4 = torch.cat((self.L_l_e4, 1/idx_data.shape[0]*torch.sum(self.L_l_e4_batch).unsqueeze(-1)), dim=0)          #entire domain
                self.L_l_e5 = torch.cat((self.L_l_e5, 1/idx_data.shape[0]*torch.sum(self.L_l_e5_batch).unsqueeze(-1)), dim=0)          #entire domain


                if algo == 'progressive_spatial' or algo == 'progressive_spatial_temporal' or algo == 'bc' or algo == 'bc_ic':
                    L_l_data_plot = 1/idx_bc.shape[0]*(torch.sum(self.L_l_uz_batch) + torch.sum(self.L_l_ur_batch) + torch.sum(self.L_l_uth_batch) + torch.sum(self.L_l_p_batch)) + 1/idx_data.shape[0]*torch.sum(self.L_l_temp_batch) #bc #bc #bc #bc #entire domain
                elif algo == 'temp_only':
                    L_l_data_plot = 1/idx_data.shape[0]*torch.sum(self.L_l_temp_batch) #entire domain
                
                L_l_eq_plot = 1/idx_data.shape[0]*(torch.sum(self.L_l_e1_batch) + torch.sum(self.L_l_e2_batch) + torch.sum(self.L_l_e3_batch) + torch.sum(self.L_l_e4_batch) + torch.sum(self.L_l_e5_batch))                       #entire domain
                L_l_tot_plot = self.alpha*L_l_data_plot + (1-self.alpha)*L_l_eq_plot
                
                self.L_l_data = torch.cat((self.L_l_data, self.alpha*L_l_data_plot.unsqueeze(-1)), dim=0)
                self.L_l_eq = torch.cat((self.L_l_eq, (1-self.alpha)*L_l_eq_plot.unsqueeze(-1)), dim=0)
                self.L_l_tot = torch.cat((self.L_l_tot, L_l_tot_plot.unsqueeze(-1)), dim=0)

            with torch.no_grad():
                # print loss values
                if (self.epoch) % 1 == 0:
                    print("it.", self.epoch.item(), "L_data", self.L_l_data[-1].item(), "L_eq", self.L_l_eq[-1].item(), "L_tot", self.L_l_tot[-1].item()) 
                
                if (self.epoch) % save_every == 0:
                    print('Saving model parameters...')
                    self.save_checkpoint(save_path=self.folder_model + self.folder_name + '.pth')
                    self.save_checkpoint_model(save_path=self.folder_model + self.folder_name + 'epoch_' + str(self.epoch.item()) + '.pth')

                    # plot losses and errors and save plots in directories
                    print('Plotting loss and error...')
                    self.plot_l_ei(dir_path=self.folder_loss)
                    self.plot_l_data(dir_path=self.folder_loss, algo=algo)
                    self.plot_loss(dir_path=self.folder_loss)
                    self.plot_error(dir_path=self.folder_error, rel_err=True)
                    self.plot_error(dir_path=self.folder_error, rel_err=False)
                    self.plot_err(dir_path=self.folder_error, rel_err=True)
                    self.plot_err(dir_path=self.folder_error, rel_err=False)

                # store validation_epochs & corresponding errors in tensors
                if (self.epoch) % eval_every == 0:
#                     print('Evaluating train...')
                    self.eval_train(b_size_eval=b_size_eval, input_tensor=input_tensor, v_data=v_data, p_data=p_data, temp_data=temp_data, idx_train=idx_data, rel_err=True, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0)
                    self.eval_train(b_size_eval=b_size_eval, input_tensor=input_tensor, v_data=v_data, p_data=p_data, temp_data=temp_data, idx_train=idx_data, rel_err=False, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0)
                    self.list_epoch_eval = torch.cat((self.list_epoch_eval, self.epoch.unsqueeze(-1)), dim=0)


        ### plot losses and errors and save plots in directories
        print('Plotting loss and error...')
        self.plot_l_ei(dir_path=self.folder_loss)
        self.plot_l_data(dir_path=self.folder_loss, algo=algo)
        self.plot_loss(dir_path=self.folder_loss)
        self.plot_error(dir_path=self.folder_error, rel_err=True)
        self.plot_error(dir_path=self.folder_error, rel_err=False)
        self.plot_err(dir_path=self.folder_error, rel_err=True)
        self.plot_err(dir_path=self.folder_error, rel_err=False)
        
        ### save model parameters
        print('Saving model parameters...')
        self.save_checkpoint(save_path=self.folder_model + self.folder_name + '.pth')
        
        print('... simulation terminated!')


    # plot results on rz-plane or rth-plane
    def eval_results(self, input_tensor: torch.Tensor, temp_data: torch.Tensor, p_data: torch.Tensor, v_data: torch.Tensor, z_data: torch.Tensor, th_data: torch.Tensor, d_0: int or float, a_0: int or float, rho_0: int or float, temp_0: int or float):
        print('Plotting results...')
        # first time step
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='z', t_last=False, save=True, save_dir=self.folder_scalar, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"temperature": temp_data, "pressure": p_data})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='th', t_last=False, save=True, save_dir=self.folder_scalar, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"temperature": temp_data, "pressure": p_data})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='z', t_last=False, save=True, save_dir=self.folder_vector, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"uz": v_data[:,0:1], "ur": v_data[:,1:2], "uth": v_data[:,2:3]})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='th', t_last=False, save=True, save_dir=self.folder_vector, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"uz": v_data[:,0:1], "ur": v_data[:,1:2], "uth": v_data[:,2:3]})
        # last time step
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='z', t_last=True, save=True, save_dir=self.folder_scalar, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"temperature": temp_data, "pressure": p_data})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='th', t_last=True, save=True, save_dir=self.folder_scalar, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"temperature": temp_data, "pressure": p_data})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='z', t_last=True, save=True, save_dir=self.folder_vector, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"uz": v_data[:,0:1], "ur": v_data[:,1:2], "uth": v_data[:,2:3]})
        gt_est_err_plane(alpha=self.alpha, input_tensor=input_tensor, z_data=z_data, th_data=th_data, model=self.model, plane='th', t_last=True, save=True, save_dir=self.folder_vector, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, **{"uz": v_data[:,0:1], "ur": v_data[:,1:2], "uth": v_data[:,2:3]})
        print('...terminated!')






