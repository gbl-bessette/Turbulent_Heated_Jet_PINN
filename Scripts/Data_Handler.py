import torch
from typing import Tuple


# Shuffle training set
def shuffle_idx_train(idx_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # get number of data points in input tensor
    N = idx_tensor.shape[0]
    
    # Shuffle the indices of the input tensor
    indices = torch.randperm(N)
    return indices 


# Add gaussian noise to any tensor in args
def add_gaussian_noise(noise_level, *args):
    mean = 0.
    std = 1.
    out = ()
    for tensor in args:
        if tensor.shape[-1] == 1:
            noise = noise_level*torch.std(tensor)*torch.normal(mean, std, size=tensor.size())
            noisy_tensor = tensor + noise
        else:
            noisy_tensor = tensor.detach().clone()
            for i in range(tensor.shape[-1]):
                noise = noise_level*torch.std(tensor[:,i])*torch.normal(mean, std, size=tensor[:,i].size())
                noisy_tensor[:,i] = tensor[:,i] + noise
        out = out + (noisy_tensor, )
    return out


# Compute relative error of noisy targets wrt. ground truth
def rel_err_noise(temp_data: torch.Tensor, temp_data_noisy: torch.Tensor, p_data: torch.Tensor, p_data_noisy: torch.Tensor, v_data: torch.Tensor, v_data_noisy: torch.Tensor):
    rel_err_noise_temp = torch.mean(torch.square((temp_data_noisy - temp_data) / torch.std(temp_data))) * 100
    rel_err_noise_p = torch.mean(torch.square((p_data_noisy - p_data) / torch.std(p_data))) * 100
    rel_err_noise_uz = torch.mean(torch.square((v_data_noisy[:,0] - v_data[:,0]) / torch.std(v_data[:,0]))) * 100
    rel_err_noise_ur = torch.mean(torch.square((v_data_noisy[:,1] - v_data[:,1]) / torch.std(v_data[:,1]))) * 100
    rel_err_noise_uth = torch.mean(torch.square((v_data_noisy[:,2] - v_data[:,2]) / torch.std(v_data[:,2]))) * 100
    min_rel_err_noise = torch.stack((rel_err_noise_uz, rel_err_noise_ur, rel_err_noise_uth, rel_err_noise_p, rel_err_noise_temp))
    return min_rel_err_noise



# Define indices for corresponding spatial and temporal zones for the progressive spatial and progressive spatio-temporal algorithms
def get_idx_zone_bc(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):
    if time_zones == True:
        bool_not_bc = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) #* (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc == 0) * (input_tensor[:, 3] <= t_data[k])).squeeze(-1)
    else:
        bool_not_bc = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) #* (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc == 0)).squeeze(-1)
    return idx_zone


def get_idx_zone_bc_ic(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):
    if time_zones == True:
        bool_not_bc_ic = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) * (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc_ic == 0) * (input_tensor[:, 3] <= t_data[k])).squeeze(-1)
    else:
        bool_not_bc_ic = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) * (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc_ic == 0)).squeeze(-1)
    return idx_zone


def get_idx_zone_0(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):
    if time_zones == True:
        bool_not_bc = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) * (input_tensor[:, 3] >= t_data[k]) #* (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc == 0) * (input_tensor[:, 3] <= t_data[k])).squeeze(-1)
    else:
        bool_not_bc = (input_tensor[:, 0] > z_data[0]) * (input_tensor[:, 0] < z_data[-1]) * (input_tensor[:, 1] > r_data[0]) * (input_tensor[:, 1] < r_data[-1]) * (input_tensor[:, 2] > th_data[0]) * (input_tensor[:, 2] < th_data[-1]) #* (input_tensor[:, 3] > t_data[0])
        idx_zone = torch.nonzero((bool_not_bc == 0)).squeeze(-1)
    return idx_zone


def get_idx_zone_1(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):

    factor_1 = (2/3)**(1/3)

    z = z_data[-1] - z_data[0]
    r = r_data[-1] - r_data[0]
    th = th_data[-1] - th_data[0]

    z_max_1 = (1 + factor_1) / 2 * z + z_data[0]
    z_min_1 = (1 - factor_1) / 2 * z + z_data[0]

    r_max_1 = (1 + factor_1) / 2 * r + r_data[0]
    r_min_1 = (1 - factor_1) / 2 * r + r_data[0]

    th_max_1 = (1 + factor_1) / 2 * th + th_data[0]
    th_min_1 = (1 - factor_1) / 2 * th + th_data[0]

    if time_zones == True:
        bool_not_zone = (input_tensor[:, 0] > z_min_1) * (input_tensor[:, 0] < z_max_1) * (input_tensor[:, 1] > r_min_1) * (input_tensor[:, 1] < r_max_1) * (input_tensor[:, 2] > th_min_1) * (input_tensor[:, 2] < th_max_1) * (input_tensor[:, 3] >= t_data[k])
        idx_zone = torch.nonzero((bool_not_zone == 0) * (input_tensor[:, 3] <= t_data[k])).squeeze(-1)
    else:
        bool_not_zone = (input_tensor[:, 0] > z_min_1) * (input_tensor[:, 0] < z_max_1) * (input_tensor[:, 1] > r_min_1) * (input_tensor[:, 1] < r_max_1) * (input_tensor[:, 2] > th_min_1) * (input_tensor[:, 2] < th_max_1)
        idx_zone = torch.nonzero((bool_not_zone == 0)).squeeze(-1)
    return idx_zone


def get_idx_zone_2(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):

    factor_2 = (1/3)**(1/3)

    z = z_data[-1] - z_data[0]
    r = r_data[-1] - r_data[0]
    th = th_data[-1] - th_data[0]

    z_max_2 = (1 + factor_2) / 2 * z + z_data[0]
    z_min_2 = (1 - factor_2) / 2 * z + z_data[0]

    r_max_2 = (1 + factor_2) / 2 * r + r_data[0]
    r_min_2 = (1 - factor_2) / 2 * r + r_data[0]

    th_max_2 = (1 + factor_2) / 2 * th + th_data[0]
    th_min_2 = (1 - factor_2) / 2 * th + th_data[0]

    if time_zones == True:
        bool_not_zone = (input_tensor[:, 0] > z_min_2) * (input_tensor[:, 0] < z_max_2) * (input_tensor[:, 1] > r_min_2) * (input_tensor[:, 1] < r_max_2) * (input_tensor[:, 2] > th_min_2) * (input_tensor[:, 2] < th_max_2) * (input_tensor[:, 3] >= t_data[k])
        idx_zone = torch.nonzero((bool_not_zone == 0) * (input_tensor[:, 3] <= t_data[k])).squeeze(-1)
    else:
        bool_not_zone = (input_tensor[:, 0] > z_min_2) * (input_tensor[:, 0] < z_max_2) * (input_tensor[:, 1] > r_min_2) * (input_tensor[:, 1] < r_max_2) * (input_tensor[:, 2] > th_min_2) * (input_tensor[:, 2] < th_max_2)
        idx_zone = torch.nonzero((bool_not_zone == 0)).squeeze(-1)
    return idx_zone


def get_idx_zone_3(input_tensor, z_data, r_data, th_data, t_data, k, time_zones: bool):
    if time_zones == True:
        bool_not_zone = (input_tensor[:, 3] > t_data[k])
        idx_zone = torch.nonzero((bool_not_zone == 0)).squeeze(-1)
    else:
        idx_zone = torch.nonzero(torch.ones_like(input_tensor[:,0])).squeeze(-1)
    return idx_zone


def get_idx_zone(input_tensor, idx_train, z_data, r_data, th_data, t_data, zone, k, time_zones: bool, condition: str):
    input_tensor_train = input_tensor[idx_train,:]
    if condition == 'bc':
        idx_colloc = get_idx_zone_bc(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
    elif condition == 'bc_ic':
        idx_colloc = get_idx_zone_bc_ic(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
    elif condition == 'data':
        if zone == 0:
            idx_colloc = get_idx_zone_0(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
        if zone == 1:
            idx_colloc = get_idx_zone_1(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
        if zone == 2:
            idx_colloc = get_idx_zone_2(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
        if zone == 3:
            idx_colloc = get_idx_zone_3(input_tensor_train, z_data, r_data, th_data, t_data, k, time_zones)
    return idx_colloc



