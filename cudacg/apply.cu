#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"CG.h"




__global__ void gpu_apply_kernel(const N *n, const float*x, float *y){//(const N *n, const float* x, float* y){

 
  int ix, iy, iz;
  //grid spacings in all directions
  float hx = 1./n->x;   
  float hy = 1./n->y;
  float hz = 1./n->z;
  float hx_inv2 = 1./(hx*hx);
  float hy_inv2 = 1./(hy*hy);
  float hz_inv2 = 1./(hz*hz);

    
  ix=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  iy=blockIdx.y*BLOCK_SIZE+threadIdx.y;
  iz=blockIdx.z*BLOCK_SIZE+threadIdx.z;

  // Diagonal element
  y[GLINIDX(n, ix,iy,iz)]=(delta+2.0*omega2 * (hx_inv2 + hy_inv2 + lambda2*hz_inv2))* x[GLINIDX(n, ix,iy,iz)];
  // Off diagonal elements, enforce homogenous Dirichlet
  // boundary conditions 
  if (ix>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix-1,iy,iz)]* (-omega2*hx_inv2);
  if (ix<n->x-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix+1,iy,iz)]* (-omega2*hx_inv2);
  if (iy>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy-1,iz)]* (-omega2*hy_inv2);
  if (iy<n->y-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy+1,iz)]* (-omega2*hy_inv2);
  if (iz>0)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy,iz-1)]* (-omega2*lambda2*hz_inv2);
  if (iz<n->z-1)
    y[GLINIDX(n, ix,iy,iz)]+= x[GLINIDX(n, ix,iy,iz+1)]* (-omega2*lambda2*hz_inv2);
 
}

int gpu_apply(const N *n, const REAL *dev_x, REAL *dev_y){

  
  N *dev_n;
   
  
  cudaMalloc((void**)&dev_n,sizeof(N));


  cudaMemcpy(dev_n,n,sizeof(N),cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(n->x/BLOCK_SIZE,n->y/BLOCK_SIZE,n->z/BLOCK_SIZE);
  //dim3 dimGrid(n->x,n->y,n->z);

  gpu_apply_kernel<<<dimGrid,dimBlock>>>(dev_n,dev_x,dev_y);

    

  cudaFree(dev_n);
  //cudaFree(dev_x);
  //cudaFree(dev_y);
 
  

 

  return(0);
}
