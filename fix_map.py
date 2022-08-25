import numpy as np

MAP_NAME = '8x12_map_cropped.txt'
MAP_DIMS = (64,64)


map = np.loadtxt(MAP_NAME, dtype=float, skiprows=2)
map = np.reshape(map, MAP_DIMS)

np.savetxt(MAP_NAME, map)