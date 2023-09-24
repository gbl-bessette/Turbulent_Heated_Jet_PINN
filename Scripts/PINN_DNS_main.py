import argparse
import torch
import os
from PINN_DNS import *


#################################################################################################################################
#################################################################################################################################

# specify simulation arguments : python3 path_to_PINN_DNS_main.py --method 'run' --algo 'temp_only' --alpha 0.9 --noise_level 0.0 --num_epoch 5 --b_size 10000 --zone_every 10000 --save_every 50000 --seed 42
# simulation model and results are saved under the Results folder in the parent directory Turbulent_Heated_Jet_Flow
# simulation results include relative and absolute errors during training, loss values during training, visualisation of scalar and vectorial fields on two different planes 'rz_plane: th' and 'rtheta-plane: z', for the first and last time steps, at the end of the simulation

#################################################################################################################################
#################################################################################################################################


parser = argparse.ArgumentParser(description = 'Runs PINN simulation for specified algorithm: temp_only, bc, progressive_spatial, progressive_spatial_temporal')

# specify method: run or load_run
# run: the simulation starts from scratch using the initialisation
# load_run: a checkpoint is loaded and the simulation runs from the loaded model (useful when simulation is interrupted, or to continue simulation for a larger number of iterations)
parser.add_argument('--method', metavar='method', type=str, help='enter your running method: run, load_run')

# specify algorithm: temp_only, bc, bc_ic, progressive_spatial, progressive_spatial_temporal
# temp_only: considering only temperature targets
# bc: considering additional boundary conditions for every output variable
# bc_ic: considering initial and boundary conditions for every output variable
# progressive_spatial (experimental): solving progressively from domain boundaries to interior of domain: division of domain in 4 spatial zones
# progressive_spatial_temporal (experimental): solving progressively from domain boundaries to interior of domain, 1 time-step at a time
parser.add_argument('--algo', metavar='algo', type=str, help='enter your algorithm: temp_only, bc, bc_ic, progressive_spatial, progressive_spatial_temporal')

# specify value of alpha: 0 < float < 1
# a larger value of alpha gives more weight to the data loss, whereas a small value of alpha favors the residuals
parser.add_argument('--alpha', metavar='alpha', type=float, help='enter float: value of prefactor alpha')

# specify noise level: float
# add noise on the DNS target values
# a noise level of 0 will not affect the target data values
parser.add_argument('--noise_level', metavar='noise_level', type=float, help='enter float: specify gaussian noise level on target values')

# specify the number of epochs for the simulation
parser.add_argument('--num_epoch', metavar='num_epoch', type=int, help='enter the number of epochs for the simulation')

# specify the mini-batch size
parser.add_argument('--b_size', metavar='b_size', type=int, help='enter the size of mini-batches')

# specify the number of epochs after which the progressive algorithm will consider an additional zone (ignored for temp_only & bc)
parser.add_argument('--zone_every', metavar='zone_every', type=int, help='enter the frequency at which to update zones for progressive learning')

# specify the number of epochs after which the model will be saved
parser.add_argument('--save_every', metavar='save_every', type=int, help='enter the frequency at which the simulation will save & plot results: save every ... iterations')

# specify the seed for stochatstic processes
parser.add_argument('--seed', metavar='seed', type=int, help='enter the manual seed that will be used for simulation')

args = parser.parse_args()


### Simulation Parameters ###
if __name__ == "__main__":

    # cuda for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device = ', device)

    # Set the seed for the random number generators
    seed = args.seed
    torch.manual_seed(seed)  # Set seed for CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for GPU operations
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    
    
    # physics parameters
    Re_const = 10000
    Pr = 0.7
    d_0 = 0.015
    a_0 = 340.26
    rho_0 = 1.225
    temp_0 = 288.16

    # number of frames to consider
    num_frames = 10

    # model parameters
    input_dim = 4
    output_dim = 5
    d_model = 100
    nb_hidden_layers = 10

    # loss parameter and learning rate
    alpha = args.alpha
    lr = 1e-3

    # training parameters
    num_epoch = args.num_epoch 
    zone_every = args.zone_every
    eval_every = 1
    save_every = args.save_every
    b_size = args.b_size #34000#24192
    # b_size_ic = 9620
    # b_size_bc = 7250 #11000
    b_size_eval = 50000

    method = args.method
    algo = args.algo
    noise_level = args.noise_level

    # parent directory
    dir = os.path.dirname(os.getcwd())

    # specify path to data file
    load_data_path = dir + '/Data/final_grid_flipz_noreshape.pth'
    # specify path to results directory
    folder_dir = dir + '/Results/'
    # folder name in results
    folder_name = algo + '_seed' + str(seed) + '_alpha' + str(alpha) + '_noise_level' + str(noise_level) + '_h' + str(nb_hidden_layers) + '_d' + str(d_model)

    # path to checkpoint directory for 'load_run' method
    load_folder_dir = folder_dir
    load_folder_name = algo + '_seed' + str(seed) + '_alpha' + str(alpha) + '_noise_level' + str(noise_level) + '_h' + str(nb_hidden_layers) + '_d' + str(d_model) 
    load_path = load_folder_dir + load_folder_name + '/' + 'model' + '/' + load_folder_name + '.pth'


    ### DATA GENERATION ###
    
    # load tensors from data file
    input_tensor, z_data, r_data, th_data, t_data, temp_data, p_data, v_data, dens_data, visc_data, min_max_input, mean_output, std_output, grid_spacing = torch.load(load_data_path, map_location=torch.device('cpu')).values()

    # print input_tensor shape
    print(input_tensor.shape)

    # reshape tensors [N, num_features]
    input_tensor = input_tensor.reshape((-1, 4))
    temp_data, p_data, v_data = temp_data.reshape(-1,1), p_data.reshape(-1,1), v_data.reshape(-1,3)

    # shuffle training indices
    idx_train = shuffle_idx_train(input_tensor)
    # print number of points in DNS data
    print('shape_idx_train', idx_train.shape)

    # add noise if noise level to targets if bigger than 0
    if noise_level > 0:
        temp_data_noisy, p_data_noisy, v_data_noisy = add_gaussian_noise(noise_level, temp_data, p_data, v_data)     
    else:
        temp_data_noisy, p_data_noisy, v_data_noisy = temp_data, p_data, v_data
    
    # compute and print reference relative error for noisy targets
    min_rel_err_noise = rel_err_noise(temp_data=temp_data, temp_data_noisy=temp_data_noisy, p_data=p_data, p_data_noisy=p_data_noisy, v_data=v_data, v_data_noisy=v_data_noisy)
    print('min_rel_err_noise', min_rel_err_noise)

    
    ### TRAINING ###

    # instantiate the PINN model and initialise parameters, optimiser and scheduler
    model = PINN_Model(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nb_hidden_layers=nb_hidden_layers, criterion=nn.MSELoss(), alpha=alpha, lr=lr, folder_dir=folder_dir, folder_name=folder_name, device=device, min_max_input=min_max_input, mean_output=mean_output, std_output=std_output, noise_level=noise_level, min_rel_err_noise=min_rel_err_noise)
    model.init_param(init_method = 'Xavier')
    model.init_optimizer(optimizer = 'Adam')
    model.init_scheduler(scheduler = 'MultistepLR')

    # load checkpoint if method 'load_run'
    if method == 'run':
        print('run code from start')

    elif method == 'load_run':
        print('load model and run code')
        model.load_checkpoint(load_path=load_path)

    # train the PINN model
    model.train(num_epoch=num_epoch, zone_every=zone_every, eval_every=eval_every, save_every=save_every, b_size=b_size, b_size_eval=b_size_eval, 
                input_tensor=input_tensor, z_data=z_data, r_data=r_data, th_data=th_data, t_data=t_data, v_data=v_data, p_data=p_data, temp_data=temp_data,
                v_data_noisy=v_data_noisy, p_data_noisy=p_data_noisy, temp_data_noisy=temp_data_noisy,  
                idx_train=idx_train, Re_const=Re_const, Pr=Pr, d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0, algo=algo)
    
    # plot visualitions
    model.eval_results(input_tensor=input_tensor.to(device), temp_data=temp_data.to(device), p_data=p_data.to(device), v_data=v_data.to(device), z_data=z_data.to(device), th_data=th_data.to(device), d_0=d_0, a_0=a_0, rho_0=rho_0, temp_0=temp_0)
