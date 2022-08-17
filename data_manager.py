import os
import numpy as np
import torch
import skfmm

INPUTS_DIR = 'datasets'

# ASSUME: all maps in subset have same dimensions


def load_paths(paths_dir, map_shape=None):

    # Make sure paths directory exists
    if not os.path.isdir(paths_dir):
        return None

    # If map shape is None, create list of ndarrays containing a series of coordinates, else put the paths in matrix form
    if map_shape == None:
        loaded = []
    else:
        loaded = np.empty((0,0,0,0))

    paths = os.scandir(paths_dir)
    for item in paths:
        # Load the paths from the files
        path = np.loadtxt(item)

        if map_shape == None:
            # add the path to the list
            loaded.append(path)
        else:
            # Convert the path to a matrix (and create the endpoints matrix)
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
            
            # Create endpoints matrix
            end_mat[path[0,1],path[0,0]] = 2                                   # Set first point in the path to 2
            end_mat[path[path.shape[0]-1,1], path[path.shape[0]-1,0]] = 2      # Include the last point in the path as 2

            x = path[0,0]
            x1 = path[0,0]
            x2 = path[path.shape[0]-1,0]

            y = path[0,1]
            y1 = path[0,1]
            y2 = path[path.shape[0]-1,1]

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
            path_mat = path_mat[np.newaxis, np.newaxis,:,:]
            end_mat = end_mat[np.newaxis, np.newaxis,:,:]
            path = np.concatenate((path_mat, end_mat), axis=1)

            loaded = np.concatenate((loaded, path), axis=0)

    return loaded

def load_map(map_file, sdf=False):

    # Make sure the map file exists
    if not os.path.isfile(map_file):
        return None

    obs_map = np.loadtxt(map_file)

    # Apply signed distance function
    if sdf:
        obs_map *= -1
        obs_map[obs_map == 0] = 1
        obs_map = skfmm.distance(obs_map, dx = 0.1)
    
    obs_map = obs_map[np.newaxis, np.newaxis, :, :]

    return obs_map

def load_input(dataset, subset):
    loaded = np.empty((0,0,0,0))

    set_dir = os.path.join(os.getcwd(), INPUTS_DIR, dataset, subset) # Open subset directory
    if not os.path.isdir(set_dir):
        return None
    
    maps = os.scandir(set_dir)
    for item in maps:
        path_dir = os.path.join(item.path, 'paths')
        if os.path.isdir(path_dir):
            obs_map = load_map(os.path.join(item.path, f'{item.name}.txt'), sdf=True)
            paths = load_paths(os.path.join(item.path, 'paths'), obs_map.shape)

            obs_map = np.repeat(obs_map, paths.shape[0], axis=0)
            paths = np.concatenate((paths, obs_map), axis=1)

            loaded = np.concatenate((loaded, paths), axis=0)

    return torch.tensor(loaded)




# TODO: method for loading trained models
def load_checkpoint():
    print("TODO: load_checkpoint()")

# TODO: method for saving model class definitions
def save_gan():
    print("TODO: save_gan()")

# TODO: method for saving checkpoints
def save_checkpoint():
    print("TODO: save_checkpoint()")