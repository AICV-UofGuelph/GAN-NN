import os, math
import numpy as np
import matplotlib.pyplot as plt

def sort_data(path, start, goal):

    sorted_path = [[start[0], start[1]]]              # ensure start point is proper path start point
    remaining_points = [[x,y] for x,y in path]
    remaining_points.append([goal[0], goal[1]])

    sorted_path_len = len(sorted_path)
    while sorted_path[sorted_path_len-1] != [goal[0], goal[1]]:
        prev_x, prev_y = sorted_path[sorted_path_len-1][0], sorted_path[sorted_path_len-1][1]
        next_point_idx = None
        smallest_dis = None
        
        for i in range(len(remaining_points)):
            curr_x, curr_y = remaining_points[i][0], remaining_points[i][1]
            dis = math.sqrt((prev_x-curr_x)**2 + (prev_y-curr_y)**2)

            if smallest_dis == None or smallest_dis > dis:
                smallest_dis = dis
                next_point_idx = i

        if smallest_dis != 0:
            sorted_path.append(remaining_points[next_point_idx])
        remaining_points.pop(next_point_idx)

        sorted_path_len = len(sorted_path)

    return np.asarray(sorted_path)

def smoothen(path, loops):
    new_path = path.astype(float)

    for _ in range(loops):
        for i in range(1, len(path)-1):
            prev_x, prev_y = path[i-1][0], path[i-1][1]
            curr_x, curr_y = path[i][0], path[i][1]
            next_x, next_y = path[i+1][0], path[i+1][1]

            new_path[i] = [(prev_x+curr_x+next_x)/3.0, (prev_y+curr_y+next_y)/3.0]
            
        path = new_path

    return new_path

def smoothen_paths(paths, start, goal, smooth_val=5, figsize=(7.5,7.5), save=True, display=True):

    new_paths = []

    if save:
        dir_name = "smoothened_paths/"
        if not os.path.exists(dir_name):                                    # if path folder for this map doesn't exist, create it
            os.mkdir(dir_name)

    for i in range(len(paths)):

        # getting path data from GAN output:
        paths[i] = np.round(paths[i])
        path_coords = np.argwhere(paths[i]==1)
        path_coords = sort_data(path_coords, start, goal)
        new_path_coords = smoothen(path_coords, smooth_val)

        if display:
            # displaying path data:
            plt.figure(figsize=figsize)
            plt.imshow(paths[i])
            plt.plot(path_coords[:, 1], path_coords[:, 0], c='r')
            plt.plot(new_path_coords[:, 1], new_path_coords[:, 0], c='g', linewidth=5)
            plt.show()

        if save:
            # flatten + write path to file:
            flat_path = new_path_coords.flatten()      # flipped so start point is at front of file, end point is at end of file
            np.savetxt(f"{dir_name}path_{i}.txt", flat_path, fmt='%d') 

        new_paths.append(new_path_coords)

    return new_paths
