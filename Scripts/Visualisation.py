import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator, AutoLocator

# get indices of data points on the z_r plane located half-way through the theta dimension, at the last timestep
def get_idx_z_r_plane_mid_t_last(input_tensor: torch.Tensor, th_data):
    mid_th = th_data[len(th_data)//2]
    return torch.nonzero(((input_tensor[:, 2] == mid_th)) * (input_tensor[:, -1] >= torch.max(input_tensor[:, -1]))).reshape(-1)

# get indices of data points on the th_r plane located half-way through the z dimension, at the last timestep
def get_idx_th_r_plane_mid_t_last(input_tensor: torch.Tensor, z_data):
    mid_z = z_data[len(z_data)//2]
    return torch.nonzero(((input_tensor[:, 0] == mid_z)) * (input_tensor[:, -1] >= torch.max(input_tensor[:, -1]))).reshape(-1)

# get indices of data points on the z_r plane located half-way through the theta dimension, at the first timestep
def get_idx_z_r_plane_mid(input_tensor, th_data):
    mid_th = th_data[len(th_data)//2]
    return torch.nonzero(((input_tensor[:, 2] == mid_th)) * (input_tensor[:, -1] <= torch.min(input_tensor[:, -1]))).reshape(-1)

# get indices of data points on the th_r plane located half-way through the z dimension, at the first timestep
def get_idx_th_r_plane_mid(input_tensor, z_data):
    mid_z = z_data[len(z_data)//2]
    return torch.nonzero(((input_tensor[:, 0] == mid_z)) * (input_tensor[:, -1] <= torch.min(input_tensor[:, -1]))).reshape(-1)
    
        
# compute relative errors on the different planes defined above, for any output of the PINN
# specify a dictionary containing outputs of the neural network and corresponding name "temperature", "pressure" or "uz", "ur", "uth", 
# to display their relative error on a specific plane
            
def gt_est_err_plane(alpha: int or float, input_tensor: torch.Tensor, z_data: torch.Tensor, th_data: torch.Tensor, model: nn.Module, plane: str, t_last: bool, save: bool, save_dir: str, d_0: int or float, a_0: int or float, rho_0: int or float, temp_0: int or float, **kwargs):

    if plane == 'th':
        fig, ax = plt.subplots(len(kwargs),3,figsize=(3*4.5, 3*len(kwargs)))
        fig.tight_layout(h_pad=5, w_pad=6.6, pad=3)
        if t_last == True:
            idx_plane = get_idx_z_r_plane_mid_t_last(input_tensor, th_data)
        else: 
            idx_plane = get_idx_z_r_plane_mid(input_tensor, th_data)
        input_tensor_plane = input_tensor[idx_plane,:]
        z_plot = input_tensor_plane[:,0].reshape(-1).tolist()
        r_plot = input_tensor_plane[:,1].reshape(-1).tolist()
        fraction = 0.0475*(r_plot[-1]-r_plot[0])/(z_plot[-1]-z_plot[0])

        
    if plane == 'z':
        fig, ax = plt.subplots(len(kwargs),3,figsize=(3*5,5*len(kwargs)))
        fig.tight_layout(h_pad=6, w_pad=8, pad=3)
        if t_last == True:
            idx_plane = get_idx_th_r_plane_mid_t_last(input_tensor, z_data)
        else: 
            idx_plane = get_idx_th_r_plane_mid(input_tensor, z_data)
        input_tensor_plane = input_tensor[idx_plane,:]
        z_plot = (input_tensor_plane[:,1]*torch.cos(input_tensor_plane[:,2])).reshape(-1).tolist()
        r_plot = (input_tensor_plane[:,1]*torch.sin(input_tensor_plane[:,2])).reshape(-1).tolist()
        fraction = 0.0263*(r_plot[-1]-r_plot[0])/(z_plot[-1]-z_plot[0])

        

    out = model.forward(input_tensor[idx_plane,:])
    out_dict = {"uz": out[:,0], "ur": out[:,1], "uth": out[:,2], "pressure": out[:,3], "temperature": out[:,4]}
    str_k = '_'
    if len(kwargs) == 1:
        for i, (k,tensor) in enumerate(kwargs.items()):
            
            str_k = str_k + str(k) + '_'
            vmin = torch.min(torch.min(tensor[idx_plane,:], torch.min(out_dict[k]))).item()
            vmax = torch.max(torch.max(tensor[idx_plane,:], torch.max(out_dict[k]))).item()
            

            if k == "uz" or k == "ur" or k == "uth":
                norm = plt.Normalize(vmin=a_0*vmin, vmax=a_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(a_0*vmin, a_0*vmax)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)
                
                # plot ground truth for velocities
                gt_i = ax[0].tricontourf(z_plot, r_plot, (a_0*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=a_0*vmin, vmax=a_0*vmax,  cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm) #levels=np.linspace(a_0*vmin,a_0*vmax,11),
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[m/s]', pad=10)

                # plot estimation from PINNN for velocities
                est_i = ax[1].tricontourf(z_plot, r_plot, (a_0*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=a_0*vmin, vmax=a_0*vmax,  cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm) #levels=np.linspace(a_0*vmin,a_0*vmax,11),
                cbar_est_i = plt.colorbar(est_i, ax=ax[1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[m/s]', pad=10)

            
            elif k == "pressure":
                norm = plt.Normalize(vmin=rho_0*a_0*a_0*vmin, vmax=rho_0*a_0*a_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(rho_0*a_0*a_0*vmin/1000, rho_0*a_0*a_0*vmax/1000)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)
                
                # plot ground truth for pressure
                gt_i = ax[0].tricontourf(z_plot, r_plot, (rho_0*a_0*a_0/1000*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=rho_0*a_0*a_0*vmin/1000, vmax=rho_0*a_0*a_0*vmax/1000, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[kPa]', pad=10)
                
                # plot estimation from PINNN for pressure
                est_i = ax[1].tricontourf(z_plot, r_plot, (rho_0*a_0*a_0/1000*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=rho_0*a_0*a_0*vmin/1000, vmax=rho_0*a_0*a_0*vmax/1000, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_est_i = plt.colorbar(est_i, ax=ax[1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[kPa]', pad=10)
            
            else:
                norm = plt.Normalize(vmin=temp_0*vmin, vmax=temp_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(temp_0*vmin, temp_0*vmax)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)

                # plot ground truth for temperature
                gt_i = ax[0].tricontourf(z_plot, r_plot, (temp_0*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=temp_0*vmin, vmax=temp_0*vmax, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[K]', pad=10)
                
                # plot estimation from PINN for temperature
                est_i = ax[1].tricontourf(z_plot, r_plot, (temp_0*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=temp_0*vmin, vmax=temp_0*vmax, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_est_i = plt.colorbar(est_i, ax=ax[1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[K]', pad=10)  
            
            # plot relative error
            relative_err_list = comp_error(k, input_tensor, idx_plane, out_dict, tensor)
            vmin = np.min(relative_err_list)
            vmax = np.max(relative_err_list)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar_gt_i_locator = AutoLocator()
            ticks = cbar_gt_i_locator.tick_values(vmin, vmax)
            levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)

            err_i = ax[2].tricontourf(z_plot, r_plot, relative_err_list, levels=levels, vmin=vmin, vmax=vmax, cmap='binary', norm=norm)
            cbar_err_i = plt.colorbar(err_i, ax=ax[2], ticks=ticks, fraction=fraction)
            cbar_err_i.ax.set_title('[%]', pad=10)



            # add titles and reshape graphs
            ax[0].set_title('ground truth: ' + str(k))
            ax[1].set_title('estimated: ' + str(k) + f" (\u03B1 = {alpha:.1f})")
            ax[2].set_title('rel. error: ' + str(k) + f" (\u03B1 = {alpha:.1f})")
            

            if plane == 'th':
                ax[0].set_xlabel("[z]")
                ax[0].set_ylabel("[r]")
                ax[1].set_xlabel("[z]")
                ax[1].set_ylabel("[r]")
                ax[2].set_xlabel("[z]")
                ax[2].set_ylabel("[r]")
                
            if plane == 'z':
                ax[0].set_xlabel("[x]")
                ax[0].set_ylabel("[y]")
                ax[1].set_xlabel("[x]")
                ax[1].set_ylabel("[y]")
                ax[2].set_xlabel("[x]")
                ax[2].set_ylabel("[y]")
                
        if save==True:
            plt.savefig(save_dir + str_k + 'plane_' + plane + '.png')
            plt.close()
        else:
            pass

    else:
        for i, (k,tensor) in enumerate(kwargs.items()):
            
            str_k = str_k + str(k) + '_'
            vmin = torch.min(torch.min(tensor[idx_plane,:], torch.min(out_dict[k]))).item()
            vmax = torch.max(torch.max(tensor[idx_plane,:], torch.max(out_dict[k]))).item()
            

            if k == "uz" or k == "ur" or k == "uth":
                norm = plt.Normalize(vmin=a_0*vmin, vmax=a_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(a_0*vmin, a_0*vmax)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)
                
                # plot ground truth for velocities
                gt_i = ax[i][0].tricontourf(z_plot, r_plot, (a_0*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=a_0*vmin, vmax=a_0*vmax,  cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm) #levels=np.linspace(a_0*vmin,a_0*vmax,11),
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[i][0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[m/s]', pad=10)

                # plot estimation from PINNN for velocities
                est_i = ax[i][1].tricontourf(z_plot, r_plot, (a_0*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=a_0*vmin, vmax=a_0*vmax,  cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm) #levels=np.linspace(a_0*vmin,a_0*vmax,11),
                cbar_est_i = plt.colorbar(est_i, ax=ax[i][1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[m/s]', pad=10)

            
            elif k == "pressure":
                norm = plt.Normalize(vmin=rho_0*a_0*a_0*vmin, vmax=rho_0*a_0*a_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(rho_0*a_0*a_0*vmin/1000, rho_0*a_0*a_0*vmax/1000)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)
                
                # plot ground truth for pressure
                gt_i = ax[i][0].tricontourf(z_plot, r_plot, (rho_0*a_0*a_0/1000*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=rho_0*a_0*a_0*vmin/1000, vmax=rho_0*a_0*a_0*vmax/1000, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[i][0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[kPa]', pad=10)
                
                # plot estimation from PINNN for pressure
                est_i = ax[i][1].tricontourf(z_plot, r_plot, (rho_0*a_0*a_0/1000*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=rho_0*a_0*a_0*vmin/1000, vmax=rho_0*a_0*a_0*vmax/1000, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_est_i = plt.colorbar(est_i, ax=ax[i][1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[kPa]', pad=10)
            
            else:
                norm = plt.Normalize(vmin=temp_0*vmin, vmax=temp_0*vmax)
                cbar_gt_i_locator = AutoLocator()
                ticks = cbar_gt_i_locator.tick_values(temp_0*vmin, temp_0*vmax)
                levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)

                # plot ground truth for temperature
                gt_i = ax[i][0].tricontourf(z_plot, r_plot, (temp_0*tensor[idx_plane,:]).reshape(-1).tolist(), levels=levels, vmin=temp_0*vmin, vmax=temp_0*vmax, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_gt_i = plt.colorbar(gt_i, ax=ax[i][0], ticks=ticks, fraction=fraction)
                cbar_gt_i.ax.set_title('[K]', pad=10)
                
                # plot estimation from PINN for temperature
                est_i = ax[i][1].tricontourf(z_plot, r_plot, (temp_0*out_dict[k]).reshape(-1).tolist(), levels=levels, vmin=temp_0*vmin, vmax=temp_0*vmax, cmap=plt.cm.get_cmap('Blues').reversed(), norm=norm)
                cbar_est_i = plt.colorbar(est_i, ax=ax[i][1], ticks=ticks, fraction=fraction)
                cbar_est_i.ax.set_title('[K]', pad=10)  
            
            # plot relative error
            relative_err_list = comp_error(k, input_tensor, idx_plane, out_dict, tensor)
            vmin = np.min(relative_err_list)
            vmax = np.max(relative_err_list)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar_gt_i_locator = AutoLocator()
            ticks = cbar_gt_i_locator.tick_values(vmin, vmax)
            levels = np.linspace(ticks[0], ticks[-1], len(ticks)*2-1)

            err_i = ax[i][2].tricontourf(z_plot, r_plot, relative_err_list, levels=levels, vmin=vmin, vmax=vmax, cmap='binary', norm=norm)
            cbar_err_i = plt.colorbar(err_i, ax=ax[i][2], ticks=ticks, fraction=fraction)
            cbar_err_i.ax.set_title('[%]', pad=10)


            if k == "uz":
                ax[i][0].set_title('Ground Truth ' + r'$u_z$')
                ax[i][1].set_title('Estimated ' + r'$u_z$' + f" : \u03B1 = {alpha:.2f}")
                ax[i][2].set_title('Rel. Error ' + r'$u_z$' + f" : \u03B1 = {alpha:.2f}")

            if k == "ur":
                ax[i][0].set_title('Ground Truth ' + r'$u_r$')
                ax[i][1].set_title('Estimated ' + r'$u_r$' + f" : \u03B1 = {alpha:.2f}")
                ax[i][2].set_title('Rel. Error ' + r'$u_r$' + f" : \u03B1 = {alpha:.2f}")

            if k == "uth":
                ax[i][0].set_title('Ground Truth ' + r'$u_{\theta}$')
                ax[i][1].set_title('Estimated ' + r'$u_{\theta}$' + f" : \u03B1 = {alpha:.2f}")
                ax[i][2].set_title('Rel. Error ' + r'$u_{\theta}$' + f" : \u03B1 = {alpha:.2f}")

            if k == "pressure":
                ax[i][0].set_title('Ground Truth ' + r'$p$')
                ax[i][1].set_title('Estimated ' + r'$p$' + f" : \u03B1 = {alpha:.2f}")
                ax[i][2].set_title('Rel. Error ' + r'$p$' + f" : \u03B1 = {alpha:.2f}")
            
            if k == "temperature":
                ax[i][0].set_title('Ground Truth ' + r'$T$')
                ax[i][1].set_title('Estimated ' + r'$T$' + f" : \u03B1 = {alpha:.2f}")
                ax[i][2].set_title('Rel. Error ' + r'$T$' + f" : \u03B1 = {alpha:.2f}")
            

            if plane == 'th':
                ax[i][0].set_xlabel("[z]")
                ax[i][0].set_ylabel("[r]")
                ax[i][1].set_xlabel("[z]")
                ax[i][1].set_ylabel("[r]")
                ax[i][2].set_xlabel("[z]")
                ax[i][2].set_ylabel("[r]")
                
            if plane == 'z':
                ax[i][0].set_xlabel("[x]")
                ax[i][0].set_ylabel("[y]")
                ax[i][1].set_xlabel("[x]")
                ax[i][1].set_ylabel("[y]")
                ax[i][2].set_xlabel("[x]")
                ax[i][2].set_ylabel("[y]")

        if save==True:
            if t_last == True:
                plt.savefig(save_dir + str_k + 'plane_' + plane + '_t_last.png')
            else:
                plt.savefig(save_dir + str_k + 'plane_' + plane + '.png')
            plt.close()
        else:
            pass

# compute relative error on specified plane
def comp_error(k, input_tensor, idx_z_r_plane, out_dict, tensor_data):
    rel_err = torch.square((out_dict[k].unsqueeze(-1) - tensor_data[idx_z_r_plane]) / torch.std(tensor_data[idx_z_r_plane])) * 100
    return rel_err.reshape(-1).tolist()










