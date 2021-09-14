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
from tools import create_directory

#Select CUDA Device
useDevice = 0


#data_dir = '/home/bruno/Desktop/ssd_0/data/'
data_dir = '/home/xavier/Desktop/mhws/'
# data_dir = '/raid/bruno/data/'
# data_dir = '/data/groups/comp-astro/bruno/'
output_dir = os.path.join(data_dir, 'render_images')
create_directory( output_dir )


nFields = 1
data_parameters = { 'type': 'random', 'dims':[512, 512, 512], 'log_data': False, 'normalization':'local', 'n_border':0 }

n_image = 0


data_to_render_list = [ get_Data_to_Render( data_parameters ) for i in range(nFields)]


volumeRender.render_parameters[0] = { 'transp_type':'flat', 'colormap':{}, 'transp_center':0.5, "transp_ramp": 1, 'density':.05, "brightness":1.0, 'transfer_offset': 0.0, 'transfer_scale': 1. }
volumeRender.render_parameters[0]['colormap']['main'] = 'matplotlib'
volumeRender.render_parameters[0]['colormap']['name'] = 'jet'
volumeRender.render_parameters[0]['colormap']['type'] = 'cmocean'


#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape

#Size of the Image
image_height, image_width = 1024, 2048

#Initialize openGL
volumeRender.image_width = image_width
volumeRender.image_height = image_height  
volumeRender.nTextures = nFields
volumeRender.nWidth = nx
volumeRender.nHeight = ny
volumeRender.nDepth = nz


#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=False )
volumeRender.initCUDA()

#set thread grid for CUDA kernels
grid3D, block3D = volumeRender.get_CUDA_threads( 16, 8, 8)   #hardcoded, tune to your needs

#Initialize all gpu data
copyToScreen_list = volumeRender.Initialize_GPU_Data(  data_to_render_list ) 

# Copy the data to the gpu
for i in range(nFields):  copyToScreen_list[i]

# Create Array for the Image 
image_data_h = np.zeros( [4, image_height, image_width], dtype=np.int32 )
image_data_d = gpuarray.to_gpu( image_data_h )

# Apply a rotation of the data
rotation_angle = 20
volumeRender.viewRotation[1] = rotation_angle


# Render the Image
volumeRender.render_image( render_to=image_data_d )
image_data_h = image_data_d.get()

# Reshape the image to RGBA
rgba_data = np.zeros( [image_height, image_width, 4], dtype=np.uint8 )
for i in range(4): rgba_data[:,:, i] = image_data_h[i,:,:]

# Save Image to PNG
out_file_name = output_dir + f'render.png' 
image_rgba = Image.fromarray( rgba_data )
image_rgb = image_rgba.convert('RGB')
image_rgb.save( out_file_name )
print(f'Saved Image: {out_file_name}' )

