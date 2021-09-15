import numpy as np
import h5py as h5
from load_data_cholla import load_snapshot_data_distributed


global_data_parameters = {}

parameters_dm_50Mpc = { 'transp_type':'sigmoid', 'cmap_indx':0, 'transp_center':0, "transp_ramp": 2.5, 'density':0.03, "brightness":2.0, 'transfer_offset': 0, 'transfer_scale': 1 }

def Interpolate_Data( data_0, data_1, index, n_indices, print_out=True, precision='float32' ):
  data_0 = data_0['data']
  data_1 = data_1['data']
  data_type_0 = data_0.dtype
  data_type_1 = data_1.dtype
  if data_type_0 != data_type_1:
    print( 'ERROR: Data sets have different data types' )
    return None
  if index == 0:
    data_interp = data_0
  elif index == n_indices:
    data_interp = data_1
  else:
    if precision == 'float32':  alpha = np.float32( index / n_indices )
    if precision == 'float16':  alpha = np.float16( index / n_indices )
    data_interp =  data_0 + alpha * ( data_1 - data_0 )
  # print( data_interp.dtype )
  return {'data':data_interp}


def get_Data_to_Render( data_parameters, print_out=True, prepare=True, output_uint=True,  precision='float32', skewers=False, back_border=True, box_ratio=1  ):
  data_dic = get_data( data_parameters, print_out=print_out, precision=precision  )
  if prepare:
    plotData = prepare_data( data_dic, data_parameters, print_out=print_out, output_uint=output_uint, skewers=skewers, back_border=back_border, box_ratio=box_ratio )
    return plotData
  else:
    return data_dic
  
def load_hdf5_file( data_params ):
  file_name = data_params['file_name']
  data_keys = data_params['data_keys']
  print( f'Loading File: {file_name} '  )
  print( f' Data Keys: {data_keys} '  )
  file = h5.File( file_name, 'r' )
  data_set = file
  for key in data_keys:
    data_set = data_set[key]
  data = data_set[...]
  file.close()
  return data

def load_prepared_data( nSnap, inDir, data_parameters ):
  format = data_parameters['data_format']
  type = data_parameters['data_type']
  field = data_parameters['data_field']
  inFileName = inDir + '{0}_{1}_{2}.h5'.format( type, field, nSnap )
  inFile = h5.File( inFileName, 'r')
  data = inFile[field][...]
  return data

def get_data( data_parameters, print_out=True, precision='float32'  ):
  global global_parameters
  type = data_parameters['type']
  data_out = {}
  if type == 'field':
    data_out['data'] = data_parameters['data']
  if type == 'file_hdf5':
    data = load_hdf5_file( data_parameters )
    data_out['data'] = data
  if type == 'color_bar':
    size = data_parameters['size']
    nx, ny =  size
    nz = np.min( [nx, ny] )
    data_x   = np.linspace(0, 1, nx )
    data_xy  = np.array([ data_x for i in range( ny ) ])
    data_xyz = np.array([ data_xy for i in range( ny ) ]) 
    data_out['data'] = data_xyz
  if type == 'random':
    nx, ny, nz = data_parameters['dims']
    print( f'Generating Random Array. Size: [{nx}, {ny}, {nz}]')
    data = np.abs(np.random.rand( nx, ny, nz ))
    data_out['data'] = data
  if type == 'border':
    dim = data_parameters['dims']
    data = np.zeros( dim, dtype=np.float32 ) 
    data_out['data'] = data
  if type == 'skewers': 
    data_skewers = get_skewers_data( data_parameters )
    data_out['data'] = data_skewers
  if type == 'cholla':
    file_type = data_parameters['file_type']
    data_type = data_parameters['data_type']
    field = data_parameters['field']
    box_size = data_parameters['box_size']
    grid_size = data_parameters['grid_size']
    n_snap  = data_parameters['n_snap']
    input_dir = data_parameters['input_dir']
    if 'proc_grid' in data_parameters: proc_grid = data_parameters['proc_grid']
    else:proc_grid = None
    if file_type == 'distributed':
      if 'subgrid' in data_parameters: subgrid = data_parameters['subgrid']
      else: subgrid = None
      if precision == 'float32': prec = np.float32
      if precision == 'float16': prec = np.float16
      data_snap = load_snapshot_data_distributed( data_type, [field],  n_snap, input_dir,  box_size, grid_size,  prec, proc_grid=proc_grid, get_statistics=True, print_fields=False, print_out=print_out, subgrid=subgrid )
      data_field = data_snap[field]
      data_out['data'] = data_field
      data_out['statistics'] = data_snap['statistics'][field]
  if 'extend_data' in data_parameters:
    data = data_out['data']
    factor_x = data_parameters['extend_data']['factor_x']
    nz, ny, nx = data.shape 
    n_extend = int(nx * factor_x)
    print( f' Extending data by: {factor_x}   n_extend: {n_extend} ')
    nx_new = nx + 2*n_extend
    data_new = np.zeros( (nz, ny, nx_new) )
    data_new[:,:, n_extend:nx+n_extend] = data
    data_new[:,:, :n_extend]  = data[:,:, -n_extend:]
    data_new[:,:, -n_extend:] = data[:,:, :n_extend] 
    data_out['data'] = data_new
  return data_out
      

def prepare_data( data_dic,  data_parameters, stats=None, print_out=True, output_uint=True, skewers=False, back_border=True, box_ratio=1 ):
  if print_out: print( 'Preparing data...')
  normalize = data_parameters['normalization']
  log = data_parameters['log_data']
  n_border = data_parameters['n_border']
  
  if 'max_uint' in data_parameters: max_uint = data_parameters['max_uint']
  else: max_uint = 255    
  
  if 'min_uint' in data_parameters: min_uint = data_parameters['min_uint']
  else: min_uint = 0    
  
  plotData = data_dic['data']
  if 'statistics' in data_dic: 
    data_min = data_dic['statistics']['min']
    data_max = data_dic['statistics']['max']
  
  
  if 'sqrt_data' in data_parameters:
    if data_parameters['sqrt_data']:
      if print_out: print( ' Computing  sqrt( data ) ')
      plotData = np.sqrt(plotData) 
      data_max = plotData.max()
      data_min = plotData.min()
  print( f' Native   min:{plotData.min()}   max:{plotData.max()}  ' )
  if 'min_data' in data_parameters: data_min = data_parameters['min_data']
  else: data_min = plotData.min()
  if 'max_data' in data_parameters: data_max = data_parameters['max_data'] 
  else: data_max = plotData.max()
    
  if 'power_data' in data_parameters:
    power = data_parameters['power_data']
    print( f' Taking data^{power}' )
    plotData = plotData**power
    data_max = data_max**power
    data_min = data_min**power
    print( f'After Power  data    max:{plotData.max()}  min:{plotData.min()} ' )
    print( f'After Power  limits  max:{data_max}  min:{data_min} ' )
    
  if 'cut_max_factor' in data_parameters:
    cut_max_factor = data_parameters['cut_max_factor']
    if cut_max_factor != 1.0:
      cut_max = data_max / cut_max_factor
      if print_out: print( f' Cuting data  max_factor: {cut_max_factor}  cut:{cut_max:.4e}')
      plotData[plotData > cut_max] = cut_max 
    
  if 'clip_max_factcor' in data_parameters:
    clip_max_factcor = data_parameters['clip_max_factcor']
    clip_max = data_max / clip_max_factcor
    if print_out: print( f' Clipping data  max_factor: {clip_max_factcor}  clip:{clip_max:.4e}')
    plotData = np.clip( plotData, None, clip_max )
    data_max = clip_max
    
  if normalize == 'local':
    plotData -= data_min
    plotData[plotData < 0 ] = 0
    norm_val =  (data_max - data_min )
    if print_out: print( f' Normalizing Local:   min:{data_min}   max:{data_max}')
    if log: 
      if print_out: print( ' Computing log10( data + 1 ) ')
      plotData = np.log10( plotData + 1)
      norm_val = np.log10( norm_val + 1 )
    if print_out: print( ' Normalizing data')
    plotData = plotData / norm_val
    plotData[ plotData > 1 ] = 1
    
  if normalize == 'global':
    max_global = stats['max_global']
    min_global = stats['min_global']
    if print_out: print( "Global min:{0}    max{1}".format( min_global, max_global) )
    max_all = max_global - min_global
    plotData -= min_global
    # if log :
    #   log_max =  np.log10( max_all + 1)
    #   plotData = np.log10(plotData + 1)
    #   plotData /= log_max
  
  if print_out: print( ' Converting data to uint')
  plotData_h_256 = max_uint * plotData + min_uint
  if n_border > 0: 
    border_val = 255
    if 'border_val' in data_parameters: border_val = data_parameters['border_val']
    plotData_h_256 = set_frame(plotData_h_256, n_border, border_val, back_border=back_border, box_ratio=box_ratio )
  if skewers: plotData_h_256 = set_skewers( plotData_h_256, data_parameters )
  if output_uint: plotData_h_256 = plotData_h_256.astype(np.uint8)
  
  if 'slice' in data_parameters:
    start, depth = data_parameters['slice']['start'], data_parameters['slice']['depth']
    print( f' Selecting Slice: start: {start}  depth: {depth}  ')
    data = plotData_h_256
    nz, ny, nx = data.shape
    start = start
    end = start + depth 
    data_slice = data[ start:end, :, :].copy()
    min, max = data_slice.min(), data_slice.max()
    print( f' Slice min: {min}  max: {max} ')
    data = np.zeros_like( data )
    data[ :depth, :, :] = data_slice
    # data[ -depth:, :, :] = data_slice
    plotData_h_256 = data
    
  return plotData_h_256



def set_frame( data, n, val, box_ratio=1, back_border=True ):
  nx, ny, nz = data.shape
  n_off = 0
  if box_ratio == 2:
    n_off = nx//4

  print( f' Setting Border n={n} val={val} shape:{data.shape}  max:{data.max()}  min:{data.min()}')
  data[:,:n+n_off,:n] = val
  data[:n+n_off,:,:n] = val
  data[:n+n_off,:n+n_off,:] = val
  data[-(n+n_off):,:,:n] = val
  data[:,-(n+n_off):,:n] = val
  data[:n+n_off,:,-n:] = val
  data[:n+n_off,-(n+n_off):,:] = val
  if back_border:
    data[:,:n+n_off,-n:] = val
    data[:,-(n+n_off):,-n:] = val
    data[-(n+n_off):,:n+n_off,:] = val
    data[-(n+n_off):,-(n+n_off):,:] = val
    data[-(n+n_off):,:,-n:] = val
  return data
  

def set_skewers( data, data_parameters ):
  nx, ny, nz = data.shape
  w = data_parameters['skewer_width']
  skewer_coords = data_parameters['skewer_coords']
  skewer_val = data_parameters['skewer_val']
  off_i, off_j = nx//2, ny//2
  for coord in skewer_coords:
    coord_i, coord_j = coord
    indx_i = coord_i + off_i
    indx_j = coord_j + off_j
    data[indx_i-w: indx_i+w, indx_j-w: indx_j+w, : ] = skewer_val
  return data
      
  
  
  
  
  

def get_skewers_data( data_parameters ):
  dims = data_parameters['dims']
  skewer_val = data_parameters['skewer_val']
  skewer_width = data_parameters['skewer_width']
  skewer_coords = data_parameters['skewer_coords']
  data_skewers = np.zeros( dims, dtype=np.float32 )
  nx, ny, nz = dims
  offset_i, offset_j = nx//2, ny//2
  for skewers_pos in skewer_coords:
    pos_i, pos_j = skewers_pos
    indx_i = pos_i + offset_i
    indx_j = pos_j + offset_j
    data_skewers[indx_i-skewer_width:indx_i+skewer_width, indx_j-skewer_width:indx_j+skewer_width, : ] = skewer_val 
  return data_skewers
  


# def get_Data_for_Interpolation( nSnap, inDir, data_parameters, n_snapshots, data_for_interpolation=None,  ):
#   if data_for_interpolation == None: 
#     data_for_interpolation = {}
#     nSnap_0 = nSnap
#     nSnap_1 = nSnap_0 + 1
#     print(" Lodading Snapshot: {0}".format(nSnap_0) )
#     data_dic_0 = get_data( nSnap_0, inDir, data_parameters, stats=True )
#     data_0 = data_dic_0['data']
#     stats = data_dic_0['stats']
#     print(" Lodading Snapshot: {0}".format(nSnap_1) )
#     data_dic_1 = get_data( nSnap_1, inDir, data_parameters, stats=False )
#     data_1 = data_dic_1['data']
#     data_for_interpolation['stats'] = stats
#     data_for_interpolation['nSnap'] = nSnap_0
#     data_for_interpolation[0] = data_0
#     data_for_interpolation[1] = data_1
#     data_for_interpolation['z'] = {}
#     data_for_interpolation['z'][0] = data_dic_0['current_z']
#     data_for_interpolation['z'][1] = data_dic_1['current_z']
#   else:
#     nSnap_prev = data_for_interpolation['nSnap']
#     if nSnap - nSnap_prev != 1: print( 'ERROR: Interpolation snapshot sequence')
#     nSnap_0 = nSnap
#     nSnap_1 = nSnap_0 + 1
#     data_for_interpolation['nSnap'] = nSnap_0
#     print( " Swaping data snapshot: {0}".format(nSnap_0) )
#     data_for_interpolation[0] = data_for_interpolation[1].copy()
#     if nSnap_1 == n_snapshots:
#       print( "Exiting: Interpolation") 
#       data_1 = data_for_interpolation[0]
#     else:
#       print(" Lodading Snapshot: {0}".format(nSnap_1) )
#       data_dic_1 = get_data( nSnap_1, inDir, data_parameters, stats=False )
#       data_1 = data_dic_1['data']
# 
#       data_for_interpolation['z'][0] = data_for_interpolation['z'][1]
#       data_for_interpolation['z'][1] = data_dic_1['current_z']
#     data_for_interpolation[1] = data_1
#   return data_for_interpolation
# 
# def get_Data_List_to_Render_Interpolation( nSnap, inDir, nFields, current_frame, frames_per_snapshot, data_parameters, data_for_interpolation, n_snapshots ):
#   if data_for_interpolation == None:
#     data_for_interpolation = {}
#     for i in range(nFields):
#       data_for_interpolation[i] = None
# 
#   data_to_render_list = []
#   for i in range( nFields ):
#     if current_frame % frames_per_snapshot == 0:
#       data_for_interpolation[i] = get_Data_for_Interpolation( nSnap, inDir, data_parameters[i], n_snapshots, data_for_interpolation=data_for_interpolation[i]  )
#     stats = data_for_interpolation[i]['stats']
#     data_interpolated, z_interpolated = Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation[i] )
#     plotData = prepare_data( data_interpolated, data_parameters[0], stats=stats )
#     data_to_render_list.append( plotData )
#   return data_to_render_list, data_for_interpolation, z_interpolated
# 
# 
# 
# def Interpolate_Data( current_frame, frames_per_snapshot, data_for_interpolation ):
#   nSnap = data_for_interpolation['nSnap']
#   data_0 = data_for_interpolation[0]
#   data_1 = data_for_interpolation[1]
#   z_0 = data_for_interpolation['z'][0]
#   z_1 = data_for_interpolation['z'][1]
#   a_0 = 1. / (z_0 + 1)
#   a_1 = 1. / (z_1 + 1)
#   alpha = float( current_frame % frames_per_snapshot ) / frames_per_snapshot
#   print( ' Interpolating Snapshots  {0} -> {1}   alpha:{2}'.format( nSnap, nSnap+1, alpha) )
#   if alpha >= 1: print('ERROR: Interpolation alpha >= 1')
#   data_interpolated = data_0 + alpha * ( data_1 - data_0 )
#   a_interpolated = a_0 + alpha * ( a_1 - a_0 )
#   z_interpolated = 1./a_interpolated - 1
#   return data_interpolated, z_interpolated