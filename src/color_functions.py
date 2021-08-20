import numpy as np
import matplotlib.colors as cl
import matplotlib.cm as cm
# from turbo_cmap import *

saved_tranfer_finction = False

colorMaps_matplotlib = [ 'inferno', 'plasma', 'magma', 'viridis', 'jet', 'nipy_spectral', 'CMRmap', 'bone', 'hot', 'copper', 'turbo']
colorMaps_cmocean = [ 'deep', 'deep_r',  'dense_r', 'haline', 'haline_r', 'matter_r', 'thermal', 'ice']
colorMaps_cmocean_div = [ 'balance']
colorMaps_scientific = [ 'davos', 'devon', 'imola', 'lapaz', 'lapaz_r', 'nuuk', 'oslo', 'oslo_r', 'GrayC', 'la_jolla']
colorMaps_scientific_div = [ 'roma', 'vik'  ]
colorMaps_colorbrewer = [ 'oranges' ]
colorMaps_colorbrewer_div = [ 'red_blue', 'spectral' ]
colorMaps_cartocolors = [ 'geyser', 'temps' ]
colorMaps_light_div = [ 'red_yellow_blue', 'blue_orange_12', 'blue_orange_8' ]


cmaps_paletable = []
for cmaps in [ colorMaps_cmocean, colorMaps_cmocean_div, colorMaps_scientific, colorMaps_colorbrewer, colorMaps_cartocolors ]:
  cmaps_paletable.extend( cmaps )


availble_colormaps = { 'matplotlib':{}, 'palettable':{}}

for c in colorMaps_matplotlib:
  availble_colormaps['matplotlib'][c] = {}
  availble_colormaps['matplotlib'][c]['color_type'] = None
  
for c in colorMaps_cmocean:
  availble_colormaps['palettable'][c] = {}
  availble_colormaps['palettable'][c]['color_type'] = 'cmocean'
  
for c in colorMaps_scientific:
  availble_colormaps['palettable'][c] = {}
  availble_colormaps['palettable'][c]['color_type'] = 'scientific'


def get_Colormap_Type( colorMap ):
  color_type = None
  if colorMap in colorMaps_matplotlib: colorMap_type = 'matplotlib'
  if colorMap in colorMaps_cmocean: colorMap_type, color_type = 'palettable', 'cmocean'
  if colorMap in colorMaps_cmocean_div: colorMap_type, color_type = 'palettable', 'cmocean_div'
  if colorMap in colorMaps_scientific: colorMap_type, color_type = 'palettable', 'scientific'
  if colorMap in colorMaps_scientific_div: colorMap_type, color_type = 'palettable', 'scientific_div'
  if colorMap in colorMaps_colorbrewer: colorMap_type, color_type = 'palettable', 'colorbrewer'
  if colorMap in colorMaps_colorbrewer_div: colorMap_type, color_type = 'palettable', 'colorbrewer_div'
  if colorMap in colorMaps_cartocolors: colorMap_type, color_type = 'palettable', 'cartocolors'
  if colorMap in colorMaps_light_div: colorMap_type, color_type = 'palettable', 'light_div'
  return colorMap_type, color_type
  
  
  
#Transfer Functions
def sigmoid( x, center, ramp ):
  return 1./( 1 + np.exp(-ramp*(x-center)))

def gaussian( x, center, ramp ):
  return np.exp(-(x-center)*(x-center)/ramp/ramp)
  
  
def get_transfer_function( fig_indx, render_parameters, print_out=True, border_color=None, bit_colors=None ):
  global changed_colormap, transfer_function_set, saved_tranfer_finction
  # if transfer_function_set: return
  
  colorMap_name = render_parameters['colormap']
  
  # colorMap_type = 'matplotlib'
  # if colorMap_name in cmaps_paletable: colorMap_type = 'palettable'
  colorMap_type, color_type = get_Colormap_Type( colorMap_name )
  
    
  nSamples = 256
  # nSamples *= 64 #10-bit color 
  colorData = get_color_data_from_colormap( colorMap_name, nSamples, )

  # Set Transparency
  transp_type = render_parameters['transp_type']
  if transp_type == 'sigmoid':
    transp_vals = np.linspace(-1,1,nSamples)
    transp_center = render_parameters['transp_center']
    transp_ramp   = render_parameters['transp_ramp']
    transparency = sigmoid( transp_vals, transp_center, transp_ramp )**2
  if transp_type == 'gaussian':
    transparency = gaussian( transp_vals, transp_center, transp_ramp )
  if transp_type == 'flat': 
    transparency =  np.ones_like( nSamples ) * 0.9
  if transp_type == 'linear':
    if 'zero_point' in render_parameters:
      x_vals = np.linspace(0,1,nSamples)
      x_min, x_max = 0, 1
      zero_point = render_parameters['zero_point']
      transp_max = render_parameters['transp_max']      
      transp_min = render_parameters['transp_min']
      delta_x = x_max - zero_point
      slope = ( transp_max  ) / delta_x 
      transp_vals =   slope*( x_vals - zero_point )
      delta_x = zero_point - x_min
      slope = ( transp_min  ) / delta_x 
      transp_vals[x_vals<=zero_point] =   slope*( x_vals[x_vals<=zero_point] - zero_point )
      transparency = np.abs( transp_vals)
      
    else:
      x_min, x_max = 0, 1
      delta_x = x_max - x_min
      transp_min = render_parameters['transp_min']
      transp_max = render_parameters['transp_max']
      slope = ( transp_max  - transp_min ) / delta_x 
      x_vals = np.linspace(0,1,nSamples)
      transp_vals =  transp_min + slope*( x_vals - x_min )
      transp_vals[x_vals<x_min] = transp_min
      transp_vals[x_vals>x_max] = transp_max
      transparency = transp_vals**1.5
  if transp_type == 'steps':
    steps_x    = render_parameters['transp_steps']
    steps_vals = np.array(render_parameters['transp_vals'])
    steps_vals /= steps_vals.max()
    n = len( steps_vals )
    x_vals = np.linspace(0,1,nSamples)
    transparency = np.zeros_like( x_vals )
    for i in range(n):
      x_min, x_max = steps_x[i], steps_x[i+1]
      indices = ( x_vals >= x_min ) * ( x_vals <= x_max ) 
      transparency[indices] = steps_vals[i]
      
  
  # colorData[:,3] = (colorVals)**2
  colorData[:,3] = (transparency )
  if border_color == 'white': colorData[-1,:] = np.array([ 1.0, 1.0, 1.0, 1.0 ]) #white frame
  if border_color == 'black': colorData[-1,:] = np.array([ 0.0, 0.0, 0.0, 1.0 ])
  
  if bit_colors:
    for i in bit_colors:
      color = bit_colors[i]
      colorData[i,:] = color
  
  if 'output_transfer' in render_parameters and not saved_tranfer_finction:
    import matplotlib.pyplot as plt
    output_dir = render_parameters['output_transfer']
    
    nrows, ncols = 2, 1 
    fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,4*nrows), sharex=True )
    
    y_min = transparency.min() * 0.9
    y_max = transparency.max() * 1.1
    
    ax = ax_l[0]
    ax.plot( transparency )
    ax.set_ylabel( 'Transparency' )
    ax.set_xlim( 0, 255  )
    ax.set_ylim( y_min, y_max  )
    
    ax = ax_l[1]
    colormap = get_colormap( colorMap_name, colorMap_type=colorMap_type, color_type=color_type )
    c = [ np.linspace( 0, 255, nSamples ) for i in range( 32 ) ]
    ax.imshow( c, cmap=colormap ) 
    ax.set_xlabel( 'bin' )
    ax.set_ylabel( 'Colormap' )
    ax.set_xlim( 0, 255  )
    
    plt.subplots_adjust( hspace = 0.0, wspace=0.0)
    figure_name = output_dir + 'transfer_function.png'
    fig.savefig( figure_name, bbox_inches='tight', dpi=300 )
    print( f'Saved Transfer Function: {figure_name}')
    saved_tranfer_finction = True
  return colorData    
      
      
    
def get_colormap( colorMap, colorMap_type='matplotlib', color_type=None ):
  
  if colorMap_type == 'matplotlib':
    return colorMap

  if colorMap_type == 'palettable':
    if color_type == 'cmocean':
      import palettable.cmocean.sequential as colors
    if color_type == 'cmocean_div':
      import palettable.cmocean.diverging as colors
    if color_type == 'scientific':
      import palettable.scientific.sequential as colors
    if color_type == 'scientific_div':
      import palettable.scientific.diverging as colors
    if color_type == 'colorbrewer':
      import palettable.colorbrewer.sequential as colors
    if color_type == 'colorbrewer_div':
      import palettable.colorbrewer.diverging as colors
    if color_type == 'cartocolors':
      import palettable.cartocolors.diverging as colors
    if color_type == 'light_div':
      import palettable.lightbartlein.diverging as colors
      
    
    if colorMap == 'deep_r':  colorMap = colors.Deep_20_r
    if colorMap == 'deep':     colorMap = colors.Deep_20
    if colorMap == 'dense_r':  colorMap = colors.Dense_20_r
    if colorMap == 'haline':  colorMap = colors.Haline_20
    if colorMap == 'haline_r':  colorMap = colors.Haline_20_r
    if colorMap == 'matter_r':  colorMap = colors.Matter_20_r
    if colorMap == 'thermal':  colorMap = colors.Thermal_20
    if colorMap == 'ice':  colorMap = colors.Ice_20
    
    
    if colorMap == 'davos': colorMap = colors.Davos_20
    if colorMap == 'devon': colorMap = colors.Devon_20
    if colorMap == 'imola': colorMap = colors.Imola_20
    if colorMap == 'lapaz': colorMap = colors.LaPaz_20
    if colorMap == 'lapaz_r': colorMap = colors.LaPaz_20_r
    if colorMap == 'nuuk': colorMap = colors.Nuuk_20
    if colorMap == 'oslo': colorMap = colors.Oslo_20
    if colorMap == 'oslo_r': colorMap = colors.Oslo_20_r
    if colorMap == 'GrayC': colorMap = colors.GrayC_20
    if colorMap == 'la_jolla':  colorMap = colors.LaJolla_20_r
    
    if colorMap == 'oranges': colorMap = colors.Oranges_9
    
    if colorMap == 'balance': colorMap = colors.Balance_20
    
    if colorMap == 'geyser': colorMap = colors.Geyser_7
    if colorMap == 'temps': colorMap = colors.Temps_7
    
    if colorMap == 'red_blue': colorMap = colors.RdBu_11_r
    if colorMap == 'spectral': colorMap = colors.Spectral_11_r
    
    if colorMap == 'red_yellow_blue': colorMap = colors.RedYellowBlue_11_r
    if colorMap == 'blue_orange_8': colorMap = colors.BlueOrange8_8
    if colorMap == 'blue_orange_12': colorMap = colors.BlueOrange12_12
    
    if colorMap == 'roma': colorMap = colors.Roma_20_r
    if colorMap == 'vik': colorMap = colors.Vik_20
    
    
    return colorMap.mpl_colormap
    


def get_color_data_from_colormap( colorMap, nSamples, color_type=None ):
  
  colorVals = np.linspace(0,1,nSamples)
  
  colorMap_type, color_type = get_Colormap_Type( colorMap )

  if colorMap_type == 'matplotlib':
    # print( 'Colormap: {colorMap} from Matplotlib')
    norm = cl.Normalize(vmin=0, vmax=1, clip=False)
    cmap = cm.ScalarMappable( norm=norm, cmap=colorMap)
    colorData = cmap.to_rgba(colorVals).astype(np.float32)

  if colorMap_type == 'palettable':
    colormap = get_colormap( colorMap, colorMap_type=colorMap_type, color_type=color_type )
    
    colorData = colormap( colorVals ).astype(np.float32)

  return colorData



