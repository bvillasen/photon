

#include <cutil_inline.h>
#include <cutil_math.h>


typedef unsigned int  uint;
typedef unsigned char uchar;


extern "C"{
  
  
 
__global__ void mask_data( uchar *d_output, int nx, int ny, int nz, uchar mask_val ){


  int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
  int tid_z = blockIdx.z*blockDim.z + threadIdx.z;
  if  ( tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;
  int tid = tid_x + tid_y*nx + tid_z*nx*ny;
  
  if ( tid_z > nz/2 ) d_output[tid] = mask_val;
   
}  
  

__global__ void shift_data( uchar* d_input, uchar *d_output, long long int nx, long long int ny, long long int nz, long long int n_shift, int shift_axis ){

  long long int tid_x, tid_y, tid_z, tid_axis_shift;
  long long int tid, tid_shift;
  tid_x = blockIdx.x*blockDim.x + threadIdx.x;
  tid_y = blockIdx.y*blockDim.y + threadIdx.y;
  tid_z = blockIdx.z*blockDim.z + threadIdx.z;
  if  ( tid_x >= nx || tid_y >= ny || tid_z >= nz ) return;
  tid = tid_x + tid_y*nx + tid_z*nx*ny;
  // if ( tid == 0 ) printf("Dims: %d %d %d  \n", nx,  ny, nz );
  
  if (shift_axis == 2){
    tid_axis_shift = tid_z + n_shift;
    if ( tid_axis_shift <  0  ) tid_axis_shift += nz;
    if ( tid_axis_shift >= nz ) tid_axis_shift -= nz;
    tid_shift = tid_x + tid_y*nx + tid_axis_shift*nx*ny;
  }
  // if ( tid_shift > nx*ny*nz || tid_shift < 0 ) printf("Invalid tid_shift: %d \n", tid_shift );
  d_output[tid_shift] = d_input[tid];
 
 
}    
  
  
}