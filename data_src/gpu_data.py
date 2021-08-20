#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, time, os
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray

#Add Modules from other directories
current_directory = os.getcwd()
src_dir = current_directory + '/src/'
data_src_dir = current_directory + '/data_src/'
cuda_dir = current_directory + '/cuda_files/'

sys.path.extend( [src_dir, data_src_dir ] )
from cudaTools import np3DtoCudaArray, np2DtoCudaArray
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
mask_data_kernel = None
shift_data_kernel = None


def Init_GPU_Data( print_out=True):
  global mask_data_kernel, shift_data_kernel
  if print_out: print( 'Compiling GPU Data Kernels')
  cudaCodeFile = open(data_src_dir + "data_kernels.cu","r")
  cudaCodeString = cudaCodeFile.read()
  cudaCodeStringComplete = cudaCodeString
  cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True, include_dirs=[ cuda_dir ] )  
  mask_data_kernel = cudaCode.get_function("mask_data")
  shift_data_kernel = cudaCode.get_function("shift_data")



def get_CUDA_threads_3D( nx, ny, nz, block_size_x=8, block_size_y=8, block_size_z=8 ):
  gridx = (nx - 1) // block_size_x + 1 
  gridy = (ny - 1) // block_size_y + 1 
  gridz = (nz - 1) // block_size_z + 1 
  block3D = (block_size_x, block_size_y, block_size_z)
  grid3D = (gridx, gridy, gridz)
  # print( f'CUDA Block Size: {block3D}')
  # print( f'CUDA Grid  Size: {grid3D}')
  return grid3D, block3D
  

def mask_data( data_d, mask_val ):
  dtype = data_d.dtype
  nx, ny, nz = data_d.shape
  grid3D, block3D = get_CUDA_threads_3D( nx, ny, nz )
  mask_data_kernel( data_d, np.int32(nx),  np.int32(ny),  np.int32(nz), np.uint8( mask_val),  grid=grid3D, block=block3D )
  

def shift_data( input_data_d, output_data_d, n_shift, shift_axis=2, print_out=False ):
  nx, ny, nz = input_data_d.shape
  grid3D, block3D = get_CUDA_threads_3D( nx, ny, nz )
  if print_out: print( f' Shifting Data: {n_shift}' )
  shift_data_kernel( input_data_d, output_data_d, np.int64(nx),  np.int64(ny),  np.int64(nz), np.int64( n_shift), np.int32( shift_axis),  grid=grid3D, block=block3D )
  cuda.Context.synchronize()




def Initialize_GPU_Data(  data_to_render_list, volumeRender, create_gpuarray=False, fixed_gpuarrays=False, print_out=True ):
  if print_out: print( "\nInitializing Data")
  initialMemory = getFreeMemory( show=print_out )
  nFields = len(data_to_render_list)
  copyToScreen_list = []
  plotData_list = []
  gpu_array_list = []
  fixed_gpuarray_list = []
  if create_gpuarray:
    for i in range(nFields):
      if fixed_gpuarrays: 
        fixed_gpu_array = gpuarray.to_gpu( data_to_render_list[i] )    
        fixed_gpuarray_list.append(fixed_gpu_array)
      gpu_array = gpuarray.to_gpu( data_to_render_list[i] )
      plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( gpu_array, allowSurfaceBind=False )
      plotData_list.append(plotData_dArray)
      copyToScreen_list.append( copyToScreenArray )
      gpu_array_list.append( gpu_array )
    volumeRender.plotData_list = plotData_list
  else:  
    for i in range(nFields):
      plotData_dArray, copyToScreenArray = np3DtoCudaArray( data_to_render_list[i] )
      plotData_list.append( plotData_dArray)
      copyToScreen_list.append( copyToScreenArray )
    volumeRender.plotData_list = plotData_list
  finalMemory = getFreeMemory( show=False )
  if print_out: print( " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6) )
  return  gpu_array_list, fixed_gpuarray_list, copyToScreen_list
