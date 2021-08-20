#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#import pyglew as glew

from PIL import Image
from PIL import ImageOps

from color_functions import get_color_data_from_colormap, availble_colormaps, get_transfer_function

#Add Modules from other directories
currentDirectory = os.getcwd()
myToolsDirectory = currentDirectory + '/src/'
volRenderDirectory = currentDirectory + '/src/'
dataSrcDir = currentDirectory + '/data_src/'
cuda_dir = currentDirectory + '/cuda_files'
sys.path.extend( [myToolsDirectory, volRenderDirectory, dataSrcDir ] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray


nWidth = 128
nHeight = 128
nDepth = 128


plotData_list = []

render_text = {} 


image_width = 512*2
image_height = 512*2

boxMin = [ -1, -1, -1 ]
boxMax = [  1,  1,  1 ]


density = 0.05
brightness = 2.0
transfer_offset = 0.0
transfer_scale = 1.0


color_first_index = 0
color_second_index = 0
changed_colormap = False

colorMap = []

render_parameters = {}


viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -3.5])

invViewMatrix_h = np.arange(12).astype(np.float32)
invViewMatrix_h_1 = np.arange(12).astype(np.float32)
transferFuncArray_d = None

#Image Parameters
scaleX = 1
scaleY = 1
scaleZ = 1
separation = 0.

#Ouput Imahe number
n_image = 0

# Cuda Parameters
block2D_GL = (32, 32, 1)
grid2D_GL = (image_width//block2D_GL[0], image_height//block2D_GL[1] )


#CUDA device variables
c_invViewMatrix = None

#CUDA Kernels
render_image_kernel = None

def get_model_view_matrix( viewTranslation, viewRotation, scaleX=1., scaleY=1., scaleZ=1,  ):
  # Matrices for the View Model
  m_pos  = np.eye( 4, 4 )
  m_scl  = np.eye( 4, 4 )
  m_rotx = np.eye( 4, 4 )
  m_roty = np.eye( 4, 4 )
  m_rotz = np.eye( 4, 4 )

  m_pos[0,3] = -viewTranslation[0]
  m_pos[1,3] = -viewTranslation[1]
  m_pos[2,3] = -viewTranslation[2]

  m_scl[0,0] = scaleX
  m_scl[1,1] = scaleY
  m_scl[2,2] = scaleZ

  theta_x =  viewRotation[0] * np.pi / 180.
  m_rotx[0,0] =  1
  m_rotx[1,1] =  np.cos(theta_x)
  m_rotx[1,2] = -np.sin(theta_x)
  m_rotx[2,1] =  np.sin(theta_x)
  m_rotx[2,2] =  np.cos(theta_x)

  theta_y =  viewRotation[1] * np.pi / 180.
  m_roty[0,0] =  np.cos(theta_y)
  m_roty[0,2] =  np.sin(theta_y)
  m_roty[1,1] =  1
  m_roty[2,0] = -np.sin(theta_y)
  m_roty[2,2] =  np.cos(theta_y)

  theta_z =  viewRotation[2] * np.pi / 180.
  m_rotz[0,0] =  np.cos(theta_z)
  m_rotz[0,1] = -np.sin(theta_z)
  m_rotz[1,0] =  np.sin(theta_z)
  m_rotz[1,1] =  np.cos(theta_z)
  m_rotz[2,2] =  1

  m_view = np.dot( m_pos, m_scl )
  m_view = np.dot( m_rotx, m_view )
  m_view = np.dot( m_roty, m_view )
  m_view = np.dot( m_rotz, m_view )
  return m_view.T





def save_image(dir='', image_name='image'):
  global n_image
  glPixelStorei(GL_PACK_ALIGNMENT, 1)
  width = nTextures * image_width
  data = glReadPixels(0, 0, width, image_height, GL_RGBA, GL_UNSIGNED_BYTE)
  print( data )
  image = Image.frombytes("RGBA", (width, image_height), data)
  image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
  image_file_name = '{0}_{1}.png'.format(image_name, n_image)
  image.save(dir+image_file_name, 'PNG')
  n_image += 1
  print( 'Image saved: {0}'.format(image_file_name))




#CUDA Textures
tex = None
transferTex = None



def render_image(  render_to=None, print_out=True, border_color=None, bit_colors=None, rescale_transparency=True ):
  global image_width, image_height, density, brightness, transferOffset, transferScale
  global block2D_GL, grid2D_GL
  global tex, transferTex
  invViewMatrix_list = [ get_invViewMatrix_image() for i in range(nTextures)]
  for fig_indx in range(nTextures):
    parameters = render_parameters[fig_indx]
    density = parameters['density']
    brightness = parameters['brightness']
    transferOffset = parameters['transfer_offset']
    transferScale = parameters['transfer_scale']
    
    plot_data = plotData_list[fig_indx]
    colorData = get_transfer_function( fig_indx, parameters, print_out=print_out, border_color=border_color, bit_colors=bit_colors )

    set_transfer_function( colorData, print_out=print_out  )
    cuda.memcpy_htod( c_invViewMatrix,  invViewMatrix_list[fig_indx])
    tex.set_array(plot_data)

    if print_out: print( 'Rendering Image')
    box_xmin, box_ymin, box_zmin = boxMin
    box_xmax, box_ymax, box_zmax = boxMax
    grid2D_GL = ( iDivUp(image_width, block2D_GL[0]) , iDivUp(image_height, block2D_GL[1]) )
    plot_border = 1
    if not border_color: plot_border = 0
    rescale_transp = 1
    if not rescale_transparency: rescale_transp = 0
    render_image_kernel( render_to,  np.int32(image_width), np.int32(image_height), np.float32(density), np.float32(brightness), np.float32(transferOffset), np.float32(transferScale), np.float32(box_xmin), np.float32(box_ymin), np.float32(box_zmin), np.float32(box_xmax), np.float32(box_ymax), np.float32(box_zmax), np.int32(plot_border), np.int32(rescale_transp), grid=grid2D_GL, block = block2D_GL, texrefs=[tex, transferTex] )
    if print_out: print( 'Finished Render')


def get_invViewMatrix_image( indx=0 ):
  invViewMatrix = np.arange(12).astype(np.float32)
  scaleX = image_width / image_height 
  model_view = get_model_view_matrix( viewTranslation, viewRotation, scaleX=scaleX, scaleY=scaleY, scaleZ=scaleZ ).reshape(16)
  # model_view = np.array([scaleX,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0,   0,  3.5, 1. ])
  if nTextures == 1: model_view *= -1
  invViewMatrix[0] = model_view[0]
  invViewMatrix[1] = model_view[4]
  invViewMatrix[2] = model_view[8]
  invViewMatrix[3] = model_view[12]
  invViewMatrix[4] = model_view[1]
  invViewMatrix[5] = model_view[5]
  invViewMatrix[6] = model_view[9]
  invViewMatrix[7] = model_view[13]
  invViewMatrix[8] = model_view[2]
  invViewMatrix[9] = model_view[6]
  invViewMatrix[10] = model_view[10]
  invViewMatrix[11] = model_view[14]
  return invViewMatrix    



def iDivUp( a, b ):
  if a%b != 0:
    return a//b + 1
  else:
    return a//b



transfer_function_set = False
def set_transfer_function( colorData, print_out=True  ):
  global changed_colormap, transfer_function_set
  # if transfer_function_set: return
  transferFunc = colorData.copy()
  transferFuncArray_d, desc = np2DtoCudaArray( transferFunc )
  transferTex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  transferTex.set_filter_mode(cuda.filter_mode.LINEAR)
  transferTex.set_address_mode(0, cuda.address_mode.CLAMP)
  transferTex.set_address_mode(1, cuda.address_mode.CLAMP)
  transferTex.set_array(transferFuncArray_d)
  if print_out: print( f'Set Tranfer Function' )
  # transfer_function_set = True


initialized_CUDA = False
def initCUDA( print_out=True):
  global plotData_dArray
  global tex, transferTex
  global transferFuncArray_d
  global c_invViewMatrix
  global render_image_kernel
  global initialized_CUDA
  #print( "Compiling CUDA code for volumeRender")
  if initialized_CUDA: return
  cudaCodeFile = open(volRenderDirectory + "CUDAvolumeRender.cu","r")
  cudaCodeString = cudaCodeFile.read()
  cudaCodeStringComplete = cudaCodeString
  cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True, include_dirs=[volRenderDirectory, cuda_dir] )
  tex = cudaCode.get_texref("tex")
  transferTex = cudaCode.get_texref("transferTex")
  c_invViewMatrix = cudaCode.get_global('c_invViewMatrix')[0]
  render_image_kernel  = cudaCode.get_function("render_image")

  # if not plotData_dArray: plotData_dArray = np3DtoCudaArray( plotData_h )
  tex.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
  tex.set_filter_mode(cuda.filter_mode.LINEAR)
  tex.set_address_mode(2, cuda.address_mode.CLAMP)
  # tex.set_address_mode(1, cuda.address_mode.CLAMP)
  # tex.set_array(plotData_dArray)
  initialized_CUDA = True
  if print_out: print( "CUDA volumeRender initialized\n")




#Read and compile CUDA code
# print( "\nCompiling CUDA code")
########################################################################
from pycuda.elementwise import ElementwiseKernel
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################


