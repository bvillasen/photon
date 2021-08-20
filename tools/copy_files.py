import sys, os, time
from shutil import copyfile




data_dir = '/home/bruno/Desktop/ssd_0/data/'
input_dir  = data_dir + 'render_images/gas_2048/fly_by_z0/images_rgb/'
output_dir = data_dir + 'render_images/gas_2048/fly_by_time_z100_0_native/'

offset = 4096

n_files = 2048
for i in range( n_files ):
  src_name = input_dir  + f'render_{i}.png'
  dst_name = output_dir + f'render_{i+offset}.png'
  copyfile(src_name, dst_name)
  


