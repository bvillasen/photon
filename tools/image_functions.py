import sys, time, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

def draw_rectange( image, edge, size, color, width, transpose=False, switch_size=False ):
    
  if switch_size: size = ( size[1], size[0] )
  y0 = edge[0] 
  x0 = edge[1]
  y1 = edge[0] + size[0]
  x1 = edge[1] + size[1]
  if transpose:
    x0 = edge[0]
    y0 = edge[1]
    x1 = edge[0] + size[0]
    y1 = edge[1] + size[1]

  draw = ImageDraw.Draw( image )
  rectangle = draw.line( [(x0, y0-width//2+1), (x0, y1+width//2-1) ], fill=color, width=width )
  rectangle = draw.line( [(x0, y0), (x1, y0) ], fill=color, width=width )
  rectangle = draw.line( [(x1, y1), (x0, y1) ], fill=color, width=width )
  rectangle = draw.line( [(x1, y1+width//2-1), (x1, y0-width//2+1) ], fill=color, width=width )
  




def draw_line( image, start, end, color, width ):
  draw = ImageDraw.Draw( image )
  line = draw.line( [start, end ], fill=color, width=width )
  

def plot_dashed_line( image, n_dashes, start, end, color, line_width ):
  x_range = np.linspace( start[0], end[0], n_dashes)
  y_range = np.linspace( start[1], end[1], n_dashes)
  draw = ImageDraw.Draw( image )
  for i in range(n_dashes):
    if i%2==1:continue
    start = (x_range[i], y_range[i] )
    end = (x_range[i+1], y_range[i+1] )
    draw.line( (start, end), fill=color, width=line_width )