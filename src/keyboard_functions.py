import os, sys
import numpy as np
import volumeRender_anim as volumeRender


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
      print( "Saving Image: {0}".format( volumeRender.n_output_image))
      volumeRender.save_image(dir=volumeRender.output_dir, image_name='image')

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

