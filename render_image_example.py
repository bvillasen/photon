import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import pycuda.gpuarray as gpuarray
import matplotlib.colors as cl
import matplotlib.cm as cm
from PIL import Image

#Add Modules from other directories
currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
import volumeRender_image as volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from data_functions import *
import gpu_data
from tools import create_directory

from IPython import embed

#Select CUDA Device
useDevice = 0


#data_dir = '/home/bruno/Desktop/ssd_0/data/'
data_dir = '/home/xavier/Desktop/mhws/'
# data_dir = '/raid/bruno/data/'
# data_dir = '/data/groups/comp-astro/bruno/'
output_dir = os.path.join(data_dir, 'render_images')
#output_dir = '/home/bruno/Desktop/'
create_directory( output_dir )

# Data from file
input_file_name = '/home/xavier/Desktop/mhws/mhws_for_bruno.h5'

file = h5.File( input_file_name, 'r' )
grid = file['grid'][...]
#embed(header='41 of image example')

n_cut = 70
grid = grid[n_cut:-n_cut,::]

n_fields_to_render = 1
data_parameters = { 
'type': 'field', 
'data': grid,
#  'dims':[512, 512, 512], 
  'log_data': False, 
  'normalization':'local', 
  'min_data': 0,  
  'n_border':1  # was 0 
}

n_image = 0
data_to_render_list = [ get_Data_to_Render(
    data_parameters ) for i in range(n_fields_to_render)]

volumeRender.render_parameters[0] = { 'transp_type':'flat',  
                                     'density':.05, 
                                     "brightness":1.2, 
                                     'transfer_offset': 0.0, 
                                     'transfer_scale': 1. }
volumeRender.render_parameters[0]['colormap'] = 'jet'
volumeRender.render_parameters[0]['transp_type'] = 'linear'  
volumeRender.render_parameters[0]['transp_min'] = 0.0  
volumeRender.render_parameters[0]['transp_max'] = 1.0  
volumeRender.render_parameters[0]['output_transfer'] = output_dir

#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape

#Size of the Image
image_height, image_width = 1024, 2048

#Initialize the volumeRender
volumeRender.image_width = image_width
volumeRender.image_height = image_height  
volumeRender.nTextures = n_fields_to_render
volumeRender.nWidth = nx
volumeRender.nHeight = ny
volumeRender.nDepth = nz

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=False )

# Initialize the Volume Render Functions
volumeRender.initCUDA()

# Initialize GPU Data Functions
gpu_data.Init_GPU_Data()

#Initialize all gpu data
gpu_array_list, gpu_array_fixed_list,  copyToScreen_list = gpu_data.Initialize_GPU_Data(  data_to_render_list, volumeRender ) 

# Copy the data to the gpu
for i in range(n_fields_to_render):  copyToScreen_list[i]

# Create Array for the Image 
image_data_h = np.zeros( [4, image_height, image_width], 
                        dtype=np.int32 )
image_data_d = gpuarray.to_gpu( image_data_h )

# Apply a rotation of the data
volumeRender.viewRotation = [-11.6, -44.4, 0. ]

# Render the Image
volumeRender.render_image( render_to=image_data_d )
image_data_h = image_data_d.get()

# Reshape the image to RGBA
rgba_data = np.zeros( [image_height, image_width, 4], 
                     dtype=np.uint8 )
for i in range(4): rgba_data[:,:, i] = image_data_h[i,:,:]

# Save Image to PNG
out_file_name = os.path.join(output_dir, f'render_image.png')
image_rgba = Image.fromarray( rgba_data )
image_rgb = image_rgba.convert('RGB')
image_rgb.save( out_file_name )
print(f'Saved Image: {out_file_name}' )

