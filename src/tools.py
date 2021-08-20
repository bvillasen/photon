import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np


def print_progress( i, n, time_start ):
  import time
  time_now = time.time()
  time = time_now - time_start
  remaining = time * ( n - i ) / i

  hrs = remaining // 3600
  min = (remaining - hrs*3600) // 60
  sec = remaining - hrs*3600 - min*60
  etr = f'{hrs:02.0f}:{min:02.0f}:{sec:02.0f}'
  progres = f'Progress:   {i}/{n}   {i/n*100:.1f}%   ETR: {etr} '
  print_line_flush (progres )

def Get_Free_Memory( print_out=False):
  import psutil
  mem = psutil.virtual_memory()
  free = mem.free / 1e9
  if print_out: print( f'Free Memory: {free:.1f} GB' )
  return free 
  
def check_if_file_exists( file_name ):
  return os.path.isfile( file_name )
  

def Load_Pickle_Directory( input_name ):
  import pickle
  print( f'Loading File: {input_name}')
  dir = pickle.load( open( input_name, 'rb' ) )
  return dir
  
def Write_Pickle_Directory( dir, output_name ):
  import pickle 
  f = open( output_name, 'wb' )
  pickle.dump( dir, f)
  print ( f'Saved File: {output_name}' )


def split_indices( indices, rank, n_procs, adjacent=False ):
  n_index_total = len(indices)
  n_proc_indices = (n_index_total-1) // n_procs + 1
  indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  if adjacent: indices_to_generate = np.array([ i + rank*n_proc_indices for i in range(n_proc_indices) ])
  else: indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  indices_to_generate = indices_to_generate[ indices_to_generate < n_index_total ]
  return indices_to_generate

def extend_path( dir=None ):
  if not dir: dir = os.getcwd()
  subDirectories = [x[0] for x in os.walk(dir) if x[0].find('.git')<0 ]
  sys.path.extend(subDirectories)


def print_mpi( text, rank, size,  mpi_comm):
  for i in range(size):
    if rank == i: print( text )
    time.sleep( 0.01 )
    mpi_comm.Barrier()

def print_line_flush( terminalString ):
  terminalString = '\r' + terminalString
  sys.stdout. write(terminalString)
  sys.stdout.flush() 


def create_directory( dir ):
  print(("Creating Directory: {0}".format(dir) ))
  indx = dir[:-1].rfind('/' )
  inDir = dir[:indx]
  dirName = dir[indx:].replace('/','')
  dir_list = next(os.walk(inDir))[1]
  if dirName in dir_list: print( " Directory exists")
  else:
    os.mkdir( dir )
    print( " Directory created")


def get_files_names( inDir, fileKey='',  type=None ):
  if not type: dataFiles = [f for f in listdir(inDir) if isfile(join(inDir, f)) ]
  if type=='nyx': dataFiles = [f for f in listdir(inDir) if (f.find(fileKey) >= 0 )  ]
  if type == 'cholla': dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find(fileKey) >= 0 ) ) ]
  dataFiles = np.sort( dataFiles )
  nFiles = len( dataFiles )
  # index_stride = int(dataFiles[1][len(fileKey):]) - int(dataFiles[0][len(fileKey):])
  if not type: return dataFiles
  if type == 'nyx': return dataFiles, nFiles
  if type == 'cholla': return dataFiles, nFiles
