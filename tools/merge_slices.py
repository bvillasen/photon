import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

nx, ny = 256, 256
n_split_x, n_split_y = 4, 4 
delta_x, delta_y = nx//n_split_x, ny//n_split_y


val_ld = 0
val_lu = 0
val_rd = 3
val_ru = 6

image = np.zeros( (delta_y, delta_x) )

for j in range( delta_x ):
  for i in range( delta_y ): 
    pos_x, pos_y = j, i
    indx_x, indx_y = pos_x // delta_x, pos_y//delta_y

    id_ld = ( indx_x, indx_y )
    id_lu = ( indx_x, indx_y+1)
    id_rd = ( indx_x+1, indx_y )
    id_ru = ( indx_x+1, indx_y+1) 

    x, y = pos_x % delta_x, pos_y % delta_y
    alpha_x, alpha_y = x / delta_x, y / delta_y
    diff_x_d = val_rd - val_ld 
    diff_x_u = val_ru - val_lu 
    interp_x_d = val_ld + alpha_x*diff_x_d 
    interp_x_u = val_lu + alpha_x*diff_x_u
    interp = interp_x_d + alpha_y * ( interp_x_u - interp_x_d )
    image[i, j] = interp

nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))

ax.imshow( image, origin='lower')

figure_name = '/home/bruno/Desktop/test.png'
fig.savefig( figure_name, bbox_inches='tight', dpi=300 )
print( f'Saved Figure: {figure_name}' )
