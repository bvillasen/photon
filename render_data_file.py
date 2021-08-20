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


data_dir = '/home/bruno/Desktop/ssd_0/data/'
input_dir = data_dir + 'render_images/ocean/data/'
output_dir  = data_dir + 'render_images/ocean/'

#Select CUDA Device
useDevice = 0

in_file_name = input_dir + 'mhws_for_bruno.h5'
file = h5.File( in_file_name, 'r' )
grid = file['grid'][...]
n_cut = 70
grid = grid[n_cut:-n_cut,::]
vmax, vmin = grid.max(), grid.min()
zero_point = -vmin / ( vmax - vmin )

# grid[ grid>0] = 0
# grid = - grid


nFields = 1
data_parameters = { 
 'type': 'field', 
 'data': grid,
 'log_data': False, 
 'normalization':'local',
 # 'max_data': 2.1501,
 'min_data': 0,  
 'n_border':1
}

rotation_angle = 60
n_image = 0


data_to_render_list = [ get_Data_to_Render( data_parameters ) for i in range(nFields)]


volumeRender.render_parameters[0] = { 'density':.05, "brightness":1.2, 'transfer_offset': 0.0, 'transfer_scale': 1. }
volumeRender.render_parameters[0]['colormap'] = 'jet'
volumeRender.render_parameters[0]['transp_type'] = 'linear'  
volumeRender.render_parameters[0]['transp_min'] = 0.0  
volumeRender.render_parameters[0]['transp_max'] = 1.0  
volumeRender.render_parameters[0]['output_transfer'] = output_dir
# volumeRender.render_parameters[0]['zero_point'] = zero_point
  


#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape

#Initialize openGL
volumeRender.width_GL = int( 512*3 )
volumeRender.height_GL = int( 512*3  )
volumeRender.nTextures = nFields
volumeRender.nWidth = nx
volumeRender.nHeight = ny
volumeRender.nDepth = nz
volumeRender.scaleX = 1
volumeRender.scaleY = nx / ny
volumeRender.scaleZ = nx / nz 
volumeRender.initGL()

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#set thread grid for CUDA kernels
grid3D, block3D = volumeRender.get_CUDA_threads( 16, 8, 8)   #hardcoded, tune to your needs

# Initialize GPU Data Functions
gpu_data.Init_GPU_Data()

# Initialize the Volume Render Functions
volumeRender.initCUDA()


#Initialize all gpu data
gpu_array_list, gpu_array_fixed_list,  copyToScreen_list = gpu_data.Initialize_GPU_Data(  data_to_render_list, volumeRender ) 
# copyToScreen_list = volumeRender.Initialize_GPU_Data(  data_to_render_list ) 


volumeRender.bit_colors = { 255: (255, 255, 255, 255) }

########################################################################
send_data = True
def sendToScreen( ):
  global send_data
  if send_data:
    for i in range(nFields): 
      copyToScreen_list[i]
    send_data = False 
########################################################################

def stepFunction():
  global  nSnap
  # volumeRender.render_parameters[0]['transp_center'] = volumeRender.set_transparency_center( nSnap, z)
  # print "Transparency center = {0}".format(volumeRender.render_parameters[0]['transp_center'])
  # volumeRender.Change_Rotation_Angle( rotation_angle )
  sendToScreen( )


########################################################################
#configure volumeRender functions
volumeRender.specialKeys = volumeRender.specialKeyboardFunc
volumeRender.stepFunc = stepFunction
volumeRender.keyboard = volumeRender.keyboard
#run volumeRender animation
volumeRender.animate()

