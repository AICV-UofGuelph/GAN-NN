import data_manager as dm
import os
import shutil
from tqdm import tqdm

SET_DIR = 'datasets'
SETS = ['random_20_density', 'random_30_density', 'random_40_density']
DEST_SET = 'random_multi_density'


# create dest set dir
root_dir = os.path.join(os.getcwd(), SET_DIR)
dest = os.path.join(root_dir, DEST_SET)
if os.path.isdir(dest):
    shutil.rmtree(dest)
os.makedirs(dest)

# Open each set being combined
for i in range(len(SETS)):
    src = os.path.join(root_dir, SETS[i])
    print(SETS[i])
    if os.path.isdir(src):

        # Open each subset
        subsets = os.listdir(src)
        for j in range(len(subsets)):
            print(f'   {subsets[j]}')

            # Add subset to dest if necessary, determine next map name (count number of maps for the subset)
            dest_subset = os.path.join(dest, subsets[j])
            if not os.path.isdir(dest_subset):
                os.mkdir(dest_subset)
                map_num = 0
            else:
                map_num = len(os.listdir(dest_subset))

            # open each map in src
            src_subset = os.path.join(src, subsets[j])
            maps = os.listdir(src_subset)
            for k in tqdm(range(len(maps))):

                dest_map = os.path.join(dest_subset, f'map_{map_num}')
                os.mkdir(dest_map)

                src_map = os.path.join(src_subset, maps[k])
                if os.path.isfile(os.path.join(src_map, f'{maps[k]}.txt')) and os.path.isfile(os.path.join(src_map, 'paths', 'path_0.txt')):   # make sure there is both a map file and paths
                    shutil.copy(os.path.join(src_map, f'{maps[k]}.txt'), os.path.join(dest_map, f'map_{map_num}.txt'))   # Copy map file
                    shutil.copytree(os.path.join(src_map, 'paths'), os.path.join(dest_map, 'paths'))   # Copy paths

                map_num += 1

        print()
