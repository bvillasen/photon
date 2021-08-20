import sys, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#Add Modules from other directories
cwd = os.getcwd()
root_dir = cwd[: cwd.find('render_3D')] + 'render_3D/'
src_directory = root_dir + "/src/"
sys.path.extend([ src_directory ] )
from tools import *

pc = 3.0857e16  #m
kpc = 1e3 * pc
Mpc = 1e6 * pc

yr = 3600 * 24 * 365
Myr = 1e6 * yr

cosmology = {'H0':67.66, 'Omega_M':0.3111, 'Omega_L':0.6889 }

def get_delta_t( delta_a, current_a, cosmology ):
  H0, Omega_M, Omega_L = cosmology['H0'], cosmology['Omega_M'], cosmology['Omega_L'] 
  a_dot = H0 * np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * 1000 / Mpc
  dt = delta_a / a_dot
  return dt


use_mpi = True
if use_mpi:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_procs = comm.Get_size()
else:
  rank = 0
  n_procs = 1



data_dir = '/home/bruno/Desktop/ssd_0/data/'
# input_dir  = data_dir + 'render_images/gas_2048/fly_by_time_z100_0/images_rgb/'
input_dir  = data_dir + 'render_images/gas_2048/fly_by_z0/images_rgb/'
output_dir = data_dir + 'render_images/gas_2048/fly_by_time_z100_0_timeonly/'
if rank == 0: create_directory( output_dir )

font_file = '/home/bruno/fonts/Helvetica.ttf'
fnt = ImageFont.truetype( font_file, 80)


color_blue = (0, 191, 255)
color_blue_dark = (102, 153, 255)
color_orange = (255, 153, 0)
color_white = ( 255, 255, 255 )

field = 'density'


n_frames = 2048
a_start, a_end = 1./101, 1.
a_vals = np.linspace( a_start, a_end, n_frames )
z_vals = 1./a_vals - 1


time_evolution = False


n_subsample = 10
sim_time = 0
simulation_time = [ sim_time ]
for i in range(n_frames-1):
  a_now, a_next = a_vals[i], a_vals[i+1]
  a_range = np.linspace( a_now, a_next, n_subsample )
  dt = 0 
  for j in range( n_subsample -1 ):
    a_0, a_1 = a_range[j], a_range[j+1]
    delta_a = a_1 - a_0
    a_half = 0.5*(a_1 + a_0)
    dt += get_delta_t( delta_a, a_half, cosmology )
  sim_time += dt
  simulation_time.append( sim_time )
simulation_time = np.array( simulation_time ) / Myr
   




img_text_temp = Image.new('RGBA', (600, 200), color = (255, 255, 255, 0))
text_img_temp = ImageDraw.Draw(img_text_temp)
fnt = ImageFont.truetype( font_file, 75)
text = 'Temperature'
text_img_temp.text((0,0), text, font=fnt, fill=color_white)

img_text_dens = Image.new('RGBA', (600, 200), color = (255, 255, 255, 0))
text_img_dens = ImageDraw.Draw(img_text_dens)
fnt = ImageFont.truetype( font_file, 75)
text = 'Gas Density'
text_img_dens.text((0,0), text, font=fnt, fill=color_white)

img_text_dm = Image.new('RGBA', (690, 200), color = (255, 255, 255, 0))
text_img_dm = ImageDraw.Draw(img_text_dm)
fnt = ImageFont.truetype( font_file, 75)
text = 'Dark Matter Density'
text_img_dm.text((0,0), text, font=fnt, fill=color_white)

img_text_0 = Image.new('RGBA', (600, 200), color = (255, 255, 255, 0))
text_img_0 = ImageDraw.Draw(img_text_0)
fnt = ImageFont.truetype( font_file, 75)
text = '10 cMpc/h'
text_img_0.text((0,0), text, font=fnt, fill=color_white)

if time_evolution: frame_start = 0
else: frame_start = 4096



frames = range(n_frames)
frames_to_render = split_indices( frames, rank, n_procs )

frames_to_render = [ frame for frame in frames_to_render if not check_if_file_exists(output_dir + f'render_cosmo_{frame+frame_start}.png') ]
print( frames_to_render )
plot_time = True

time_start = time.time()
n = len( frames_to_render )
for i,frame in enumerate(frames_to_render):



  if time_evolution: z_val = z_vals[frame]
  else: z_val = z_vals[-1]
  if z_val > 1: z_text = 'z = {0:.1f}'.format( z_val )
  else: z_text = 'z = {0:.2f}'.format( z_val )
  
  if time_evolution: sim_time = simulation_time[frame]
  else: sim_time = simulation_time[-1]
  if sim_time/1000 < 1: sim_time_text = r'Time = {0:.2f} Gyr'.format( sim_time/ 1000 )
  if sim_time/1000 >= 1: sim_time_text = r'Time = {0:.1f} Gyr'.format( sim_time / 1000 )
  # print( frame, sim_time_text )
  
  
  if plot_time: text = sim_time_text
  else: text= z_text 
  

  image_name = input_dir + f'render_{frame}.png'
  image = Image.open( image_name ) 
  width, height = image.size 

  img_text = Image.new('RGBA', (1000, 200), color = (255, 255, 255, 0))
  text_img = ImageDraw.Draw(img_text)
  text_img.text((100,60), text, font=fnt, fill=color_white)
  # 
  image_final = Image.new('RGB', (width, height))
  image_final.paste( image, (0,0) )
  image_final.paste( img_text, (0,0), mask=img_text )
  # if field == 'density': image_final.paste( img_text_dens, (100,2000), mask=img_text_dens )
  # if field =='temperature': image_final.paste( img_text_temp, (3200,2000), mask=img_text_temp )
  # if field =='dm': image_final.paste( img_text_dm, (3000,2000), mask=img_text_dm )

  # 
  # line = ImageDraw.Draw( image_final )
  # x_off = 3300
  # y_off = 150
  # x_len = int(2048. / 50 * 10 )
  # shape = [(x_off, y_off), (x_off+x_len, y_off)] 
  # line.line( shape, fill='white', width = 10)
  # image_final.paste( img_text_0, (x_off+25,int(y_off*0.35)), mask=img_text_0 )

  out_img_name = output_dir + f'render_cosmo_{frame + frame_start}.png'
  image_final.save(  out_img_name )
  # print( f'Saved Image: {out_img_name}')

  if rank == 0: print_progress( i+1, n, time_start )


comm.Barrier()

if rank == 0:
  all_files = [ check_if_file_exists( output_dir + f'render_{frame + frame_start}.png') for frame in range(n_frames) ]
  n_files = sum( all_files )
  print( f'Images Saved: {n_files} ')




