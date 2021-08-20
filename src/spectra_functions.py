import sys, os
import numpy as np
from scipy.special import erf
import constants_cgs as cgs
root_dir = os.path.dirname(os.getcwd()) + '/'
from tools import print_line_flush

def get_Doppler_parameter( T, chem_type='HI' ):
  if chem_type == 'HI':   M = cgs.M_p
  if chem_type == 'HeII': M = 4 * cgs.M_p
  b = np.sqrt( 2* cgs.K_b / M * T )
  return b

  
#Copy ghost cells to extend periodic boundaries   
def extend_periodic( arr, n_ghost):
  n = len(arr)
  arr_periodic = np.zeros( n + 2*n_ghost )
  arr_periodic[n_ghost:n+n_ghost] = arr
  arr_periodic[:n_ghost] = arr[-n_ghost:]
  arr_periodic[-n_ghost:] = arr[:n_ghost]
  return arr_periodic
  


def get_optical_depth_velocity( current_z, H, dr, dv, n_HI_los, vel_peculiar_los, temp_los, space='redshift', method='error_function', chem_type='HI' ):
  # Lymann Alpha Parameters
  Lya_lambda = 1.21567e-5 #cm  Rest wave length of the Lyman Alpha Transition
  f_12 = 0.416 #Oscillator strength
  Lya_sigma = np.pi * cgs.e_charge**2 / cgs.M_e / cgs.c * f_12 * Lya_lambda 
  if chem_type == 'HeII': Lya_sigma /= 4              # source: https://arxiv.org/pdf/astro-ph/9812429.pdf
  H_cgs = H * 1e5 / cgs.kpc 
  dr_cgs = dr * cgs.kpc
  
  #Extend Ghost cells for periodic boundaries
  n = len(n_HI_los)
  n_ghost = int( 0.1 * n )
  n_HI = extend_periodic( n_HI_los, n_ghost)
  vel_peculiar = extend_periodic( vel_peculiar_los, n_ghost )
  temp = extend_periodic( temp_los, n_ghost) 
  
  r_proper = (np.linspace( -n_ghost, n+n_ghost-1, n+2*n_ghost) + 0.5 )* dr
  vel_Hubble = H * r_proper * 1e5  #cm/s
  dv_Hubble = vel_Hubble[1] - vel_Hubble[0]
  
  n_points = len( n_HI )
  
  if space == 'real': velocity = vel_Hubble
  elif space == 'redshift': velocity = vel_Hubble + vel_peculiar
  else: 
    print ('ERROR: Invalid space ( Real or Redshift )')
    return None
  
  b_all = get_Doppler_parameter( temp, chem_type=chem_type ) 
  tau_los = np.zeros(n_points) #Initialize arrays of zeros for the total optical delpth along the line of sight
    
  if method=='error_function':
    #Loop over each cell
    for j in range(n_points):
      #Get  local values of the cell
      v_j = vel_Hubble[j]                      #Hubble Velocity of the cell
      y_l = ( ( v_j - 0.5 * H_cgs * dr_cgs ) - velocity ) / b_all
      y_r = ( ( v_j + 0.5 * H_cgs * dr_cgs ) - velocity ) / b_all
      tau_val = Lya_sigma  / H_cgs  * np.sum( n_HI * ( erf(y_r) - erf(y_l) ) ) / 2
      tau_los[j] = tau_val 
            
  # Trim the ghost cells from the global optical depth 
  tau_los    = tau_los[n_ghost:-n_ghost]
  vel_Hubble = vel_Hubble[n_ghost:-n_ghost] * 1e-5 # km/seg
  return tau_los, vel_Hubble



def compute_optical_depth( cosmology, box, skewer, space='redshift', method='error_function', chem_type='HI' ):
  H0 = cosmology['H0'] 
  Omega_M = cosmology['Omega_M']
  Omega_L = cosmology['Omega_L']
  current_z = cosmology['current_z']
  cosmo_h = H0 / 100.
  H0 = H0 / 1000     #km/s/kpc
  
  Lbox_x, Lbox_y, Lbox_z = box['Lbox']
  if  Lbox_x != Lbox_y: print( 'Warning: Lbox is not the same for X Y Z')
  if  Lbox_x != Lbox_z: print( 'Warning: Lbox is not the same for X Y Z')
  Lbox = Lbox_x
  
  HI_density  = skewer['HI_density']
  velocity    = skewer['velocity']
  temperature = skewer['temperature']
  if chem_type == 'HeII': HeII_density = skewer['HeII_density']
  
  
  nPoints = len( HI_density )
  
  #Proper length
  current_a = 1./(current_z + 1)
  R = current_a * Lbox / cosmo_h
  nx = nPoints
  dr = R / nx
  
  a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
  H = a_dot / current_a
  dens_HI_los = HI_density / (current_a)**3
  temp_los = temperature
  vel_los = velocity.copy() 
  
  dv = H * dr
  
  #Convert to CGS Units
  dens_HI_los *=  cgs.Msun / cgs.kpc**3 * cosmo_h**2
  n_HI_los = dens_HI_los / cgs.M_p
  vel_los_cms = vel_los * 1e5 
  dv_cms = dv * 1e5
  # print( f'dv_Hubble: {dv_cms}')
  
  if chem_type == 'HeII':
    dens_HeII_los = HeII_density / (current_a)**3
    dens_HeII_los *=  cgs.Msun / cgs.kpc**3 * cosmo_h**2
    n_HeII_los = dens_HeII_los / ( 4 * cgs.M_p )
  
  if chem_type == 'HI':   n_los = n_HI_los  
  if chem_type == 'HeII': n_los = n_HeII_los  
    
  tau, vel_Hubble = get_optical_depth_velocity( current_z,  H, dr, dv_cms, n_los, vel_los_cms, temp_los, space=space, method=method, chem_type=chem_type )
  data_out = {}
  data_out['vel_Hubble'] = vel_Hubble
  data_out['tau'] = tau
    
  return data_out
  
def Compute_Skewers_Transmitted_Flux( skewers_data, cosmology, box ):
  # Compute the Transmitted Flux along all the skewers
  n_skewers = skewers_data['density'].shape[0]
  print( '\nComputing Flux along Skewers')
  skewers_Flux = []
  for skewer_id in range(n_skewers):

    text = ' Skewer {0}/{1}    {2:.0f} %'.format(skewer_id+1, n_skewers,  float(skewer_id+1)/n_skewers*100)
    print_line_flush( text )

    skewer_data = {}  
    skewer_data['HI_density']  = skewers_data['HI_density'][skewer_id]
    skewer_data['temperature'] = skewers_data['temperature'][skewer_id]
    skewer_data['velocity']    = skewers_data['velocity'][skewer_id]

    tau_los_data = compute_optical_depth( cosmology, box, skewer_data  )
    los_vel_hubble = tau_los_data['vel_Hubble']
    los_tau = tau_los_data['tau']
    los_F = np.exp( -los_tau )
    skewers_Flux.append( los_F )
  skewers_Flux = np.array( skewers_Flux )
  Flux_mean = skewers_Flux.mean()
  vel_Hubble = los_vel_hubble
  data_out = { 'vel_Hubble':vel_Hubble, 'Flux_mean':Flux_mean, 'skewers_Flux':skewers_Flux }
  return data_out
  
  
