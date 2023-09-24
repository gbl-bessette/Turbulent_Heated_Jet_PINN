import os

# create directories for storing results and .pth models

def create_directory(dir_path):
    # If folder does not exist, then create it
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print('created folder : ' + dir_path)
    else:
        print('folder :' + dir_path + 'already exists' )


def create_directories(dir_path, folder_name):
    folder_dir = dir_path + folder_name + '/'
    folder_results = folder_dir + 'results' + '/'
    folder_model = folder_dir + 'model' + '/'
    folder_loss = folder_results + 'loss' + '/'
    folder_error = folder_results + 'error' + '/'
    folder_scalar = folder_results + 'scalar' + '/'
    folder_vector = folder_results + 'vector' + '/'
    
    create_directory(folder_dir)
    create_directory(folder_results)
    create_directory(folder_model)
    create_directory(folder_loss)
    create_directory(folder_error)
    create_directory(folder_scalar)
    create_directory(folder_vector)
    
    return folder_model, folder_loss, folder_error, folder_scalar, folder_vector