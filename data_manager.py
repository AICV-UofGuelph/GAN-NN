import os
import shutil
import importlib.util
import sys
import numpy as np
import torch
import skfmm

INPUTS_DIR = 'datasets' # Name of folder in root that contains the datasets
CHKPT_DIR = 'checkpoints' # Name of the folder in root that contains the model checkpoints, def'n, checkpoints

# ASSUME: all maps in subset have same dimensions


def load_paths(paths_dir, map_shape=None):

    # Make sure paths directory exists
    if not os.path.isdir(paths_dir):
        return None

    loaded = []

    paths = os.scandir(paths_dir)
    for item in paths:
        # Load the paths from the files
        path = np.loadtxt(item)
        if len(path.shape) == 1:
            path = np.asarray(path, dtype=int).reshape(len(path)//2,2) #unflatten the path from the file


        # if no map shape provided, load paths as arrays of coordinates
        if map_shape != None:
            path_mat = np.zeros(map_shape)
            end_mat = np.zeros(map_shape)

            # Make the path continuous
            for i in range(path.shape[0] - 1):
                x = path[i,0]
                x1 = path[i,0]
                x2 = path[i+1,0]

                y = path[i,1]
                y1 = path[i,1]
                y2 = path[i+1,1]

                if (x1 < x2):
                    x_dir = 1
                else:
                    x_dir = -1

                if (y1 < y2):
                    y_dir = 1
                else:
                    y_dir = -1

                # Determine y from x
                if x2-x1 != 0:
                    m = (y2-y1)/(x2-x1)
                    while x != x2:
                        y = round(m*(x-x1) + y1)
                        path_mat[y,x] = 1
                        x += x_dir
                else:
                    while x != x2:
                        path_mat[y1,x] = 1
                        x += x_dir


                x = path[i,0]
                x1 = path[i,0]
                x2 = path[i+1,0]

                y = path[i,1]
                y1 = path[i,1]
                y2 = path[i+1,1]

                # Determine x from y
                if y2-y1 != 0:
                    m = (x2-x1)/(y2-y1)
                    while y != y2:
                        x = round(m*(y-y1) + x1)
                        path_mat[y,x] = 1
                        y += y_dir
                else:
                    while y != y2:
                        path_mat[y,x1] = 1
                        y += y_dir
            
            # This code connects start and end point with a straight line
            # Create endpoints matrix
            end_mat[path[0,1],path[0,0]] = 2                                   # Set first point in the path to 2
            end_mat[path[path.shape[0]-1,1], path[path.shape[0]-1,0]] = 2      # Include the last point in the path as 2

            x = path[0,0]
            x1 = path[0,0] # first x
            x2 = path[path.shape[0]-1,0] # last x

            y = path[0,1]
            y1 = path[0,1] # first y
            y2 = path[path.shape[0]-1,1] # last y

            if (x1 < x2):
                x_dir = 1
            else:
                x_dir = -1

            if (y1 < y2):
                y_dir = 1
            else:
                y_dir = -1

            # Determine y from x
            if x2-x1 != 0:
                m = (y2-y1)/(x2-x1)
                while x != x2:
                    y = round(m*(x-x1) + y1)
                    if end_mat[y,x] == 0:
                        end_mat[y,x] = 1
                    x += x_dir
            else:
                while x != x2:
                    if end_mat[y1,x] == 0:
                        end_mat[y1,x] = 1
                    x += x_dir

            x = path[0,0]
            x1 = path[0,0]
            x2 = path[path.shape[0]-1,0]

            y = path[0,1]
            y1 = path[0,1]
            y2 = path[path.shape[0]-1,1]

            # Determine x from y
            if y2-y1 != 0:
                m = (x2-x1)/(y2-y1)
                while y != y2:
                    x = round(m*(y-y1) + x1)
                    if end_mat[y,x] == 0:
                        end_mat[y,x] = 1
                    y += y_dir
            else:
                while y != y2:
                    if end_mat[y,x1] == 0:
                        end_mat[y,x1] = 1
                    y += y_dir

            # Combine the two matrices
            path_mat = path_mat[np.newaxis,:,:] # matrix with real path that is now continuous
            end_mat = end_mat[np.newaxis,:,:] # end points
            path = np.concatenate((path_mat, end_mat), axis=0) # concat to give a matrix with continuous path (1 layer) and the endpoints are 2s w/ 1s connecting in straight line (other layer)

        loaded.append(path)

    return loaded

def load_map(map_file, sdf=False):

    # Make sure the map file exists
    if not os.path.isfile(map_file):
        return None

    obs_map = np.loadtxt(map_file)

    # Apply signed distance function
    if sdf:
        obs_map *= -1 # obstacles are -1
        obs_map[obs_map == 0] = 1 # ... and free space is 1
        obs_map = skfmm.distance(obs_map, dx = 0.1)
    
    obs_map = obs_map[:, :]

    return obs_map

def load_input(dataset, subset):
    loaded = []

    set_dir = os.path.join(os.getcwd(), INPUTS_DIR, dataset, subset) # Open subset directory
    if not os.path.isdir(set_dir):
        return None
    
    maps = os.scandir(set_dir)
    for item in maps:
        path_dir = os.path.join(item.path, 'paths') # for a paths folder for one map
        if os.path.isdir(path_dir):
            obs_map = load_map(os.path.join(item.path, f'{item.name}.txt'), sdf=True)
            paths = load_paths(os.path.join(item.path, 'paths'), obs_map.shape)

            for i in range(len(paths)): # for all the paths for the single map
                path = np.concatenate((paths[i], obs_map[np.newaxis, :, :]), axis=0) # adding map to path 
                path = torch.tensor(path, dtype=torch.float) # and making it a tensor
                loaded.append(path)

    return loaded


def load_gan(run_name):
    run_dir = os.path.join(os.getcwd(), CHKPT_DIR, run_name)
    if not os.path.isfile(os.path.join(run_dir, 'GAN.py')):
        return None
    elif not os.path.isfile(os.path.join(run_dir, 'hparams.pkl')):
        return None

    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location('GAN', os.path.join(run_dir, 'GAN.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['GANdefs'] = module
    spec.loader.exec_module(module)

    config = torch.load(os.path.join(run_dir, 'hparams.pkl'))
    return config

def load_checkpoint(run_name, step_num):
    cp_file = os.path.join(os.getcwd(), CHKPT_DIR, run_name, 'cp', f'{step_num}.tar')
    if not os.path.isfile(cp_file):
        return None
    
    states = torch.load(cp_file)
    return states


def save_gan(run_name, config):
    """Saves the model and config. Training notebook usually calls this once at beginning (if record metric == true)"""
    save_dir = os.path.join(os.getcwd(), CHKPT_DIR, run_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save model structure and hyperparams
    shutil.copy(os.path.join(os.getcwd(), 'GAN.py'), os.path.join(save_dir, 'GAN.py')) # will copy GAN.py from root to the save directory
    torch.save(config, os.path.join(save_dir, 'hparams.pkl')) # saves 

def save_checkpoint(run_name, step, gen, crit):
    # Navigate to destination directory
    save_path = os.path.join(os.getcwd(), CHKPT_DIR, run_name, 'cp')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f'{step}.tar')

    torch.save({
        'gen': gen.state_dict(),
        'crit': crit.state_dict()},
        save_path
    )