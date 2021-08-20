import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from shutil import copyfile




# inDir = '/home/bruno/Desktop/ssd_0/data/render_images/fly_by_2048_anim/'
inDir = '/home/bruno/Desktop/ssd_0/data/render_images/gas_2048/fly_by_time_z100_0_native/'
outDir = '/home/bruno/Desktop/'


image_name = 'render'

out_anim_name = 'fly_by_2048_time_z100_0_native'

cmd = 'ffmpeg -framerate 60  '
# cmd += ' -start_number 45'
cmd += ' -i {0}{1}_%d.png '.format( inDir, image_name )
cmd += '-pix_fmt yuv420p '
cmd += '-b 7500k '
cmd += '{0}{1}.mp4'.format( outDir, out_anim_name )
# cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
# cmd += ' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"'
os.system( cmd )

