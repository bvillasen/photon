import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np

#Add Modules from other directories
currentDirectory = os.getcwd()
srcDirectory = currentDirectory + "/src/"
dataDirectory = currentDirectory + "/data_src/"
sys.path.extend([ srcDirectory, dataDirectory ] )
import volumeRender_anim as volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from data_functions import *
import gpu_data
from keyboard_functions import keyboard, specialKeyboardFunc

output_dir  = '/home/xavier/Desktop/'

#Select CUDA Device
useDevice = 0

n_fields_to_render = 1
data_parameters = { 'type': 'random', 
 'dims':[256, 256, 256], 
 'log_data': False, 
 'normalization':'local',
 'max_data': 1.0,
 'min_data': 0.0,  
 'n_border':0 
}


data_to_render_list = [ get_Data_to_Render( data_parameters ) for i in range(n_fields_to_render)]

volumeRender.render_parameters[0] = { 'transp_type':'flat',  'density':.005, "brightness":1.1, 'transfer_offset': 0.0, 'transfer_scale': 1. }
volumeRender.render_parameters[0]['colormap'] = 'jet'

#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape

#Initialize openGL
volumeRender.width_GL  = int( 512*3 )
volumeRender.height_GL = int( 512*3 )
volumeRender.nTextures = n_fields_to_render
volumeRender.nWidth = nx
volumeRender.nHeight = ny
volumeRender.nDepth = nz
volumeRender.output_dir = output_dir
volumeRender.initGL()

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

# Initialize the Volume Render Functions
volumeRender.initCUDA()

# Initialize GPU Data Functions
gpu_data.Init_GPU_Data()

#Initialize all gpu data
gpu_array_list, gpu_array_fixed_list,  copyToScreen_list = gpu_data.Initialize_GPU_Data(  data_to_render_list, volumeRender ) 

########################################################################
copy_render_data_to_device = True
def sendToScreen( ):
  global copy_render_data_to_device
  if copy_render_data_to_device:
    for i in range(n_fields_to_render): 
      copyToScreen_list[i]
    copy_render_data_to_device = False 
########################################################################

# This function will be called each time a frame is rendered
def stepFunction():
  global  nSnap
  sendToScreen( )

########################################################################

# Configure volumeRender functions
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = stepFunction
volumeRender.keyboard = keyboard
#run volumeRender animation
volumeRender.animate()

