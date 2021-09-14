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

output_dir  = '/home/bruno/Desktop/'

#Select CUDA Device
useDevice = 0



nFields = 1
data_parameters = { 'type': 'random', 
 'dims':[256, 256, 256], 
 'log_data': False, 
 'normalization':'local',
 'max_data': 1.0,
 'min_data': 0.0,  
 'n_border':0 
}

rotation_angle = 60
n_image = 0


data_to_render_list = [ get_Data_to_Render( data_parameters ) for i in range(nFields)]


volumeRender.render_parameters[0] = { 'transp_type':'flat',  'density':.005, "brightness":1.1, 'transfer_offset': 0.0, 'transfer_scale': 1. }
volumeRender.render_parameters[0]['colormap'] = 'jet'



#Get Dimensions of the data to render
nz, ny, nx = data_to_render_list[0].shape

#Initialize openGL
volumeRender.width_GL = int( 512*3 )
volumeRender.height_GL = int( 512*3  )
volumeRender.nTextures = nFields
volumeRender.nWidth = nx
volumeRender.nHeight = ny
volumeRender.nDepth = nz
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
def specialKeyboardFunc( key, x, y ):
  global nSnap
  if key== volumeRender.GLUT_KEY_RIGHT:
    volumeRender.color_second_index += 1
    volumeRender.changed_colormap = True
  if key== volumeRender.GLUT_KEY_LEFT:
    volumeRender.color_second_index -= 1
    volumeRender.changed_colormap = True  
  if key== volumeRender.GLUT_KEY_UP:
    volumeRender.color_first_index += 1
    volumeRender.changed_colormap = True
  if key== volumeRender.GLUT_KEY_DOWN:
    volumeRender.color_first_index -= 1
    volumeRender.changed_colormap = True

  # if key== volumeRender.GLUT_KEY_RIGHT:
  #   nSnap += 1
  #   if nSnap == nSnapshots: nSnap = 0
  #   print " Snapshot: ", nSnap
  #   change_snapshot( nSnap )
# 
def keyboard(*args):
  ESCAPE = '\033'
  SPACE = '32'
  key = args[0].decode("utf-8")
  # If escape is pressed, kill everything.
  if key == ESCAPE:
    print( "Ending Render")
    #cuda.gl.Context.pop()
    sys.exit()  
  if key == 'z':
      print( "Saving Image: {0}".format( volumeRender.n_image))
      volumeRender.save_image(dir=output_dir, image_name='image')

  if key == 'q':
    volumeRender.render_parameters[0]['transp_center'] -= np.float32(0.01)
    print( "Image Transp Center: ",volumeRender.render_parameters[0]['transp_center'])
  if key == 'w':
    volumeRender.render_parameters[0]['transp_center'] += np.float32(0.01)
    print( "Image Transp Center: ",volumeRender.render_parameters[0]['transp_center'])
  if key == 'a':
    volumeRender.render_parameters[0]['transp_ramp'] -= np.float32(0.01)
    print( "Image Transp Ramp: ",volumeRender.render_parameters[0]['transp_ramp'])
  if key == 's':
    volumeRender.render_parameters[0]['transp_ramp'] += np.float32(0.01)
    print( "Image Transp Ramp: ",volumeRender.render_parameters[0]['transp_ramp'])
  if key == 'd':
    dens_min = 0.001
    volumeRender.render_parameters[0]['density'] -= np.float32(0.002)
    if volumeRender.render_parameters[0]['density'] < dens_min: 
      volumeRender.render_parameters[0]['density'] = dens_min
    print( "Image Density: ",volumeRender.render_parameters[0]['density'])
  if key == 'e':
    volumeRender.render_parameters[0]['density'] += np.float32(0.002)
    print( "Image Density: ",volumeRender.render_parameters[0]['density'])
  if key == 'f':
    volumeRender.render_parameters[0]['brightness'] -= np.float32(0.01)
    print( "Image brightness: ",volumeRender.render_parameters[0]['brightness'])
  if key == 'r':
    volumeRender.render_parameters[0]['brightness'] += np.float32(0.01)
    print( "Image brightness: ",volumeRender.render_parameters[0]['brightness'])
  if key == 't':
    volumeRender.render_parameters[0]['transfer_offset'] -= np.float32(0.01)
    print( "Image transfer_offset: ",volumeRender.render_parameters[0]['transfer_offset'])
  if key == 'g':
    volumeRender.render_parameters[0]['transfer_offset'] += np.float32(0.01)
    print( "Image transfer_offset: ",volumeRender.render_parameters[0]['transfer_offset'])
  if key == 'y':
    volumeRender.render_parameters[0]['transfer_scale'] -= np.float32(0.01)
    print( "Image transfer_scale: ",volumeRender.render_parameters[0]['transfer_scale'])
  if key == 'h':
    volumeRender.render_parameters[0]['transfer_scale'] += np.float32(0.01)
    print( "Image transfer_scale: ",volumeRender.render_parameters[0]['transfer_scale'])


  # if key == '2':
  #   transferScale -= np.float32(0.01)
  #   print "Image Transfer Scale: ",transferScale
  # if key == '4':
  #   brightness -= np.float32(0.01)

########################################################################
#configure volumeRender functions
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = stepFunction
volumeRender.keyboard = keyboard
#run volumeRender animation
volumeRender.animate()

